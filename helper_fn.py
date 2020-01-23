import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def load_df(log_dir, reset_logs):
    if not os.path.isfile(log_dir) or reset_logs:
        print("New csv file created at {}".format(log_dir))
        df = pd.DataFrame(columns=['model_name', 'epoch', 'train_loss', 'val_loss',
                                   'train_state_mse', 'val_state_mse',
                                   'train_latent_mse', 'val_latent_mse',
                                   'train_state_inf', 'val_state_inf'])
        df.to_csv(log_dir, index=False)
    else:
        print("Loading logs from {}".format(log_dir))
        df = pd.read_csv(log_dir)
    return df

def save_configs(model_name, params, path):
    if not os.path.isfile(path):
        df = pd.DataFrame(columns=['model_name', 'encoder_shape', 'n_shifts',
                           'state_loss', 'latent_loss', 'reg_loss', 'inf_loss',
                           'epochs'])
        df.to_csv(path, index=False)
    df = pd.read_csv(path)
    df.loc[len(df)+1] = [model_name, params['encoder_shape'], params['n_shifts'],
                         params['state_loss'], params['latent_loss'],
                         params['reg_loss'], params['inf_loss'], 0]
    df.to_csv(path, index=False)
    return df


def save_configs2(model_name, params, path):
    if not os.path.isfile(path):
        df = pd.DataFrame(columns=['model_name', 'encoder_shape', 'aux_shape', 'n_shifts',
                           'state_loss', 'latent_loss', 'reg_loss', 'inf_loss', 'epochs'])
        df.to_csv(path, index=False)
    df = pd.read_csv(path)
    df.loc[len(df)+1] = [model_name, params['encoder_shape'], params['aux_shape'],
                         params['n_shifts'], params['state_loss'],
                         params['latent_loss'], params['reg_loss'], params['inf_loss'], 0]
    df.to_csv(path, index=False)
    return df

def increment_epoch_config(model_name, path):
    df = pd.read_csv(path)
    df.loc[df.model_name == model_name, 'epochs'] += 1
    df.to_csv(path, index=False)


def saver(model, model_dir, df, name, val_loss = .0, condition='Best', verbose=False):
    if len(df)!=0:
        model_df = df.loc[df['model_name'] == name]
        val_loss_min = np.min(model_df['val_loss'])
    else:
        torch.save(model, model_dir)
    if condition!='Best':
        torch.save(model, model_dir)
        if verbose:
            print('Saving model at: {}'.format(model_dir))
        return
    elif val_loss <= val_loss_min:
        torch.save(model, model_dir)
        if verbose:
            print('New minimum loss found, saving model at: {}'.format(model_dir))

def plot_val(val_data, model, mse, fig_dir, num=10, figsize=(6, 4)):
    model.eval()
    plt.figure(figsize=figsize)
    val_enc_gt, val_enc_traj, _ = model(val_data)

    for i in range(10):
        n_shifts = val_enc_traj.shape[1]
        p = plt.plot(val_data[i, :n_shifts, 0], val_data[i, :n_shifts, 1], '--')
        plt.plot(val_enc_traj[i, :, 0].detach(), val_enc_traj[i, :, 1].detach(), c=p[0].get_color())
    plt.axis('off')
    plt.xlim([-3, 3])
    plt.ylim([-2, 2])
    plt.title("MSE: {:.3f}".format(mse))
    plt.savefig(fig_dir, bbox_inches='tight')
    plt.close()

def plot_val3D(val_data, model, mse, fig_dir, num=10, figsize=(10, 7)):
    model.eval()
    val_enc_gt, val_enc_traj, _ = model(val_data)
    val_enc_gt = val_enc_gt.detach()
    val_enc_traj = val_enc_traj.detach()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        n_shifts = val_enc_traj.shape[1]
        p = ax.plot(val_data[i, :n_shifts, 0].numpy(),
        val_data[i, :n_shifts, 1].numpy(),
        val_data[i, :n_shifts, 2].numpy(), '--')
        ax.plot(val_enc_traj[i, :n_shifts, 0].numpy(),
        val_enc_traj[i, :n_shifts, 1].numpy(),
        val_enc_traj[i, :n_shifts, 2].numpy(), c=p[0].get_color())
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_axis_off()
    plt.title("MSE: {:.3f}".format(mse))
    plt.savefig(fig_dir, bbox_inches='tight')
    plt.close()


def plot_layers(parameters, mse, dist_dir=None, n_rows=3):
    ps_raw = [i for i in parameters]
    n_cols = int(np.ceil(len(ps_raw)/n_rows))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

    for r in range(n_rows):
        for c in range(n_cols):
            ax[r, c].spines['right'].set_visible(False)
            ax[r, c].spines['top'].set_visible(False)
            ax[r, c].spines['left'].set_visible(False)
            ax[r, c].get_yaxis().set_visible(False)

    for idx, p_raw in enumerate(ps_raw):
        i = idx
        l = i//n_cols
        r = idx-l*n_cols
        layer = p_raw[1].detach().numpy().flatten()
        sns.kdeplot(layer, shade=True, ax=ax[l, r])
        ax[l, r].set_xlim([-1.5, 1.5])
        ax[l, r].set_ylim([0, 4])

        ax[l, r].spines['right'].set_visible(False)
        ax[l, r].spines['top'].set_visible(False)
        ax[l, r].spines['left'].set_visible(False)
        ax[l, r].get_yaxis().set_visible(True)
        ax[l, r].set_ylabel(p_raw[0], fontsize=15, labelpad=-1)
        ax[l, r].set_yticks([])

    plt.suptitle("MSE: {:.3f}".format(mse), x = 0.15, y=0.88, fontsize=15)
    plt.subplots_adjust(hspace=0.1)

    if dist_dir is not None:
        plt.savefig(dist_dir, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_gif(fig_dir, root_dir, model_name, t=10, sample=5, ext='trj'):
    images = []
    for file_name in sorted(os.listdir(fig_dir))[::sample]:
        if file_name.endswith('.png'):
            file_path = os.path.join(fig_dir, file_name)
            images.append(imageio.imread(file_path))
    save_dir = os.path.join(root_dir, 'gifs/{}-{}.gif'.format(model_name, ext))
    fps = len(images)/t
    imageio.mimsave(save_dir, images, fps=fps, format='GIF')
    print("Showing at {} fps".format(int(fps)))
    return save_dir


def plot_losses(model_df):
    fig, axes = plt.subplots(1, 4, figsize=(40, 5), dpi=100)
    axes[0].plot(model_df['train_loss'], label='train')
    axes[0].plot(model_df['val_loss'], linewidth=2, c='r', label='val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].set_yscale('log')


    axes[1].plot(model_df['train_state_mse'], label='train')
    axes[1].plot(model_df['val_state_mse'], linewidth=2, c='r', label='val')
    axes[1].set_title('State MSE')
    axes[1].legend()
    axes[1].set_yscale('log')

    axes[2].plot(model_df['train_latent_mse'], label='train')
    axes[2].plot(model_df['val_latent_mse'], linewidth=2, c='r', label='val')
    axes[2].set_title('Latent MSE')
    axes[2].legend()
    axes[2].set_yscale('log')

    axes[3].plot(model_df['train_state_inf'], label='train')
    axes[3].plot(model_df['val_state_inf'], linewidth=2, c='r', label='val')
    axes[3].set_title('Mean Max Deviation')
    axes[3].legend()
    axes[3].set_yscale('log')

    for i in range(2):
        axes[i].set_ylim([None, 1.5])
    axes[2].set_ylim([None, 0.02])
    #axes[3].set_ylim([0, None])
    plt.show()


def plot_eigen_func(ko, enc_traj, data, show=False, num=3,
                    fig_dir='eigen_funcs.png'):
    eig_fns = []
    eig_vals = []
    for idx, k in enumerate(ko):
        w, vs = np.linalg.eig(k)
        eig_vals.append(w)
        eigf = np.matmul(enc_traj[idx], vs)

        eig_fns.append(eigf)
    eig_fns = np.asarray(eig_fns)
    eig_vals = np.asarray(eig_vals)

    fig, ax = plt.subplots(num, 2, figsize=(5*num, 15))

    x = data[:, 0, 0].numpy()
    y = data[:, 0, 1].numpy()

    for i in range(num):
        z = np.abs(eig_fns[:, 2*i])
        phi = np.angle(eig_fns[:, 2*i]/eig_fns[:, 2*i+1])
        im = ax[i, 0].tricontourf(x, y, z, cmap='Purples', levels=100)
        im2 = ax[i, 1].tricontourf(x, y, phi, cmap='PuOr', levels=100)
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        #plt.colorbar(im, ax=ax[i, 0], orientation="horizontal", pad=0.1)
        #plt.colorbar(im2, ax=ax[i, 1], orientation="horizontal", pad=0.1)
    if show:
        plt.show()
    else:
        plt.savefig(fig_dir, bbox_inches='tight')
        plt.close()
