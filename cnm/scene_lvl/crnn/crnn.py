import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from cnm.scene_lvl.crnn.model import ResCNNEncoder, TwoStreamResCNNEncoder, DecoderRNN
from cnm.scene_lvl.crnn.dataset_loader import VicCycLegacyDataset
from sklearn.metrics import accuracy_score


# # Select which frame to begin & end in videos
# begin_frame, end_frame, skip_frame = 1, 29, 1


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (img_X, opt_X, y) in enumerate(train_loader):
        # distribute data to device
        img_X, opt_X, y = img_X.to(device), opt_X.to(device), y.to(device).view(-1, )
        N_count += img_X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(img_X, opt_X))   # output has dim = (batch, number of classes)

        # print(output.shape, y.shape)
        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))
        

    return losses, scores


def validation(log_interval, model, device, optimizer, test_loader, epoch, save_model_path):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (img_X, opt_X, y) in enumerate(test_loader):
            # distribute data to device
            img_X, opt_X, y = img_X.to(device), opt_X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(img_X, opt_X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

            if batch_idx % log_interval == 0:
                print('Working on testing, please wait...')

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


def main(csv_data_path, image_data_path, optical_flow_data_path,
         h_res_size=224, w_res_size=224, forward_frame_len=20, backward_frame_len=25,
         CNN_fc_hidden1=1024, CNN_fc_hidden2=768, dropout_p=0.1, CNN_embed_dim=512,
         RNN_hidden_layers=3, RNN_hidden_nodes=512, RNN_FC_dim=256, num_classes=2,
         epochs=15, batch_size=16, learning_rate=1e-3, log_interval=10, exp_data_dir=None):
    """

    :param csv_data_path:
    :param image_data_path:
    :param CNN_embed_dim: latent dim extracted by 2D CNN
    :param :
    :return:
    """
    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    # Data loading parameters
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose([transforms.Resize([h_res_size, w_res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set, valid_set = VicCycLegacyDataset(csv_data_path, image_data_path, optical_flow_data_path, split='train',
                                               forward_frame_len=forward_frame_len, backward_frame_len=backward_frame_len, transform=transform), \
                           VicCycLegacyDataset(csv_data_path, image_data_path, optical_flow_data_path, split='test',
                                               forward_frame_len=forward_frame_len, backward_frame_len=backward_frame_len, transform=transform)

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    # Create model
    cnn_encoder = TwoStreamResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                             h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=num_classes).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn_encoder = nn.DataParallel(cnn_encoder)
        rnn_decoder = nn.DataParallel(rnn_decoder)

        # Combine all EncoderCNN + DecoderRNN parameters
        crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                      list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
                      list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())
    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        # Combine all EncoderCNN + DecoderRNN parameters (ResNet is not trained)
        crnn_params = list(cnn_encoder.img_fc1.parameters()) + list(cnn_encoder.img_bn1.parameters()) + \
                      list(cnn_encoder.img_fc2.parameters()) + list(cnn_encoder.img_bn2.parameters()) + \
                      list(cnn_encoder.img_fc3.parameters()) + \
                      list(cnn_encoder.opt_fc1.parameters()) + list(cnn_encoder.opt_bn1.parameters()) + \
                      list(cnn_encoder.opt_fc2.parameters()) + list(cnn_encoder.opt_bn2.parameters()) + \
                      list(cnn_encoder.opt_fc3.parameters()) + list(rnn_decoder.parameters())

    optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)


    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    # start training
    for ep_num in range(epochs):
        # train, test model
        train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, ep_num)
        epoch_test_loss, epoch_test_score = validation(log_interval, [cnn_encoder, rnn_decoder], device, optimizer, valid_loader, ep_num,
                                                       save_model_path=exp_data_dir)

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)
        np.save(os.path.join(exp_data_dir, 'CRNN_epoch_training_losses.npy'), A)
        np.save(os.path.join(exp_data_dir, 'CRNN_epoch_training_scores.npy'), B)
        np.save(os.path.join(exp_data_dir, 'CRNN_epoch_test_loss.npy'), C)
        np.save(os.path.join(exp_data_dir, 'CRNN_epoch_test_score.npy'), D)

    # plot
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
    plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc="upper left")
    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
    plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc="upper left")
    fig_save_path = os.path.join(exp_data_dir, "fig_UCF101_ResNetCRNN.png")
    plt.savefig(fig_save_path, dpi=600)
    # plt.close(fig)
    plt.show()


if __name__ == '__main__':
    # Set default data path
    repo_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    vic_cyc_legacy_csv_path = os.path.join(repo_dir_path, 'cnm', 'dataset', 'dataset_vic_cyc_legacy', 'NearMiss_classification.csv')
    default_exp_data_dir_path = os.path.join(os.path.dirname(repo_dir_path), 'near_miss_exp_data')
    # Parsing arguments
    import argparse
    parser = argparse.ArgumentParser()
    # Dataset related arguments
    parser.add_argument('--csv_data_path', type=str, default=repo_dir_path, help='The file path of the .csv file with dataset info.')
    parser.add_argument('--image_data_path', type=str, default=None, help='The directory path saving all video frames.')
    parser.add_argument('--optical_flow_data_path', type=str, default=None, help='The directory path saving all video frames.')
    parser.add_argument('--h_res_size', type=int, default=224, help='The the height of the resized video frame.')
    parser.add_argument('--w_res_size', type=int, default=224, help='The the width of the resized video frame.')
    parser.add_argument('--forward_frame_len', type=int, default=20, help='The number of frames before the keyframe added to the image sequence')
    parser.add_argument('--backward_frame_len', type=int, default=20, help='The number of frames after the keyframe added to the image sequence') # TODO: set default to 25
    # Neural Network Model related arguments
    #   1. EncoderCNN architecture
    parser.add_argument('--CNN_fc_hidden1', type=int, default=1024, help='')
    parser.add_argument('--CNN_fc_hidden2', type=int, default=768, help='')
    parser.add_argument('--dropout_p', type=float, default=0.1, help='')
    parser.add_argument('--CNN_embed_dim', type=int, default=512, help='latent dim extracted by 2D CNN')
    #   2. DecoderRNN architecture
    parser.add_argument('--RNN_hidden_layers', type=int, default=3, help='')
    parser.add_argument('--RNN_hidden_nodes', type=int, default=512, help='')
    parser.add_argument('--RNN_FC_dim', type=int, default=256, help='')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target category')
    # Training related arguments
    parser.add_argument('--epochs', type=int, default=15, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='')
    parser.add_argument('--log_interval', type=int, default=10, help='interval for displaying training info')
    #
    parser.add_argument('--exp_name', type=str, default='crnn', help='The name of the experiment run.')
    parser.add_argument('--exp_data_dir', type=str, default=None, help='The directory to save results to.')

    args = parser.parse_args()

    # TODO: sophisticated logger to log the experiment setup and results
    # Where experiment outputs are saved by default:
    if args.image_data_path is None:
        raise ValueError("Please specify image_data_path!")

    if args.exp_data_dir is None:
        exp_data_dir = os.path.join(default_exp_data_dir_path, args.exp_name)
    else:
        exp_data_dir = os.path.join(os.path.abspath(args.exp_data_dir), args.exp_name)
    if not os.path.exists(exp_data_dir):
        os.makedirs(exp_data_dir)

    #
    main(csv_data_path=args.csv_data_path, image_data_path=args.image_data_path, optical_flow_data_path=args.optical_flow_data_path,
         h_res_size=args.h_res_size, w_res_size=args.w_res_size,
         forward_frame_len=args.forward_frame_len, backward_frame_len=args.backward_frame_len,
         CNN_fc_hidden1=args.CNN_fc_hidden1, CNN_fc_hidden2=args.CNN_fc_hidden2, dropout_p=args.dropout_p, CNN_embed_dim=args.CNN_embed_dim,
         RNN_hidden_layers=args.RNN_hidden_layers, RNN_hidden_nodes=args.RNN_hidden_nodes, RNN_FC_dim=args.RNN_FC_dim, num_classes=args.num_classes,
         epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
         log_interval=args.log_interval, exp_data_dir=exp_data_dir)