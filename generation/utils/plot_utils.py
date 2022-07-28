from matplotlib import pyplot as plt
import constants.constants as constants
from matplotlib.ticker import MaxNLocator

def plotHistLossEpoch(num_epoch, loss, t_loss=None):
    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(num_epoch+1), loss, label='loss')
    if(t_loss != None):
        ax1.plot(range(num_epoch+1), t_loss, label='test loss')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    plt.savefig(constants.saved_path+f'loss_epoch_{num_epoch}.png')
    plt.close()

def plotHistAllLossEpoch(num_epoch, loss_eye, t_loss_eye, loss_pose_r, t_loss_pose_r, loss_au, t_loss_au):
    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(num_epoch+1), loss_eye, color="darkgreen", label='Loss gaze - Train')
    ax1.plot(range(num_epoch+1), t_loss_eye, color="limegreen", label='Loss gaze - Test')

    ax1.plot(range(num_epoch+1), loss_pose_r, color="darkblue", label='Loss pose r - Train')
    ax1.plot(range(num_epoch+1), t_loss_pose_r, color="cornflowerblue", label='Loss pose r - Test')

    ax1.plot(range(num_epoch+1), loss_au, color="red", label='Loss AU - Train')
    ax1.plot(range(num_epoch+1), t_loss_au, color="lightcoral", label='Loss AU - Test')

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    plt.savefig(constants.saved_path+f'all_loss_epoch_{num_epoch}.png')
    plt.close()

def plotHistLossEpochGAN(num_epoch, d_loss, g_loss, t_loss=None):
    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(num_epoch+1), d_loss, label='discriminator loss')
    ax1.plot(range(num_epoch+1), g_loss, label='generator loss')
    if(t_loss != None):
        ax1.plot(range(num_epoch+1), t_loss, label='test loss')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    plt.savefig(constants.saved_path+f'loss_epoch_{num_epoch}.png')
    plt.close()


def plotHistPredEpochGAN(num_epoch, d_real_pred, d_fake_pred):
    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(num_epoch+1), d_real_pred, label='discriminator real prediction')
    ax1.plot(range(num_epoch+1), d_fake_pred, label='discriminator fake prediction')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Label")
    ax1.legend()
    plt.savefig(constants.saved_path+f'pred_epoch_{num_epoch}.png')
    plt.close()