
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import random

def Fileload(path):
    data = []
    with open(path, 'r') as file:
        content = file.read()
        lines = content.split("\n")
        for line in lines:
            if line.strip():
                try:
                    line = line.strip().split("\t")
                    line = [float(i) for i in line]
                    data.append(line)
                except:
                    None
    return np.asarray(data)

parser = argparse.ArgumentParser()

def save_gif(_path):

    data = Fileload(_path)

    seqs = np.unique(data[:, 0])
    ids = np.unique(data[:, 1])

    colors = [cm.rainbow(random.random()) for _ in range(len(ids))]
    id_color_map = {id_: color for id_, color in zip(ids, colors)}

    last_seq = {id_: max(data[data[:, 1] == id_][:, 0]) for id_ in ids}


    fig, ax = plt.subplots()
    scatters = {id_: ax.scatter([], [], color=id_color_map[id_]) for id_ in ids}

    ax.set_xlim(np.min(data[:, 2]), np.max(data[:, 2]))
    ax.set_ylim(np.min(data[:, 3]), np.max(data[:, 3]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Pedestrian Movement')

    history = {id_: [] for id_ in ids}

    def update(frame):
        seq_data = data[data[:, 0] == seqs[frame]]
        for id_ in ids:
            if seqs[frame] <= last_seq[id_]:
                ped_data = seq_data[seq_data[:, 1] == id_]
                if len(ped_data) > 0:
                    history[id_].append(ped_data[:, 2:4])
                    offsets = np.concatenate(history[id_], axis=0)
                    scatters[id_].set_offsets(offsets)
                    scatters[id_].set_alpha(np.linspace(0.1, 0.9, len(history[id_])))
                
            else:
                scatters[id_].set_offsets([[-10,0]]) 
        ax.set_title(f'Pedestrian Movement: Sequence {int(seqs[frame])}')
        return list(scatters.values())

    ani = animation.FuncAnimation(fig, update, frames=len(seqs), interval=150, blit=True)
    gif_path = _path.replace('.txt', '.gif').replace('/datasets/', '/visual/')
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    ani.save(gif_path, writer='ani')
    #plt.show()

def main():
    
    for root, dirs, files in os.walk('./datasets/'):
        #print(files)
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                save_gif(file_path)

if __name__ == "__main__":
    main()
