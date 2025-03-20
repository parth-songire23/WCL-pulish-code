import os
import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

class LoadAndPlot:
    """
    Load simulation data from .mat files and generate plots.
    """
    def __init__(self, store_path="./data/mannal_store/", user_num=1, attacker_num=1, RIS_ant_num=16):
        self.color_list = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
        self.store_path = store_path
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.RIS_ant_num = RIS_ant_num
        self.all_steps = self.load_all_steps()

    def load_one_ep(self, file_name):
        """Load one episode's data from a .mat file."""
        file_path = os.path.join(self.store_path, file_name)
        return loadmat(file_path)

    def load_all_steps(self, ep_num=100):
        """Load all episodes and extract relevant metrics."""
        print(f"Loading data from: {self.store_path}")

        result_dic = {
            'reward': [],
            'user_capacity': [[] for _ in range(self.user_num)],
            'secure_capacity': [[] for _ in range(self.user_num)],
            'attacker_capacity': [[] for _ in range(self.attacker_num)],
            'RIS_elements': [[] for _ in range(self.RIS_ant_num)]
        }

        for ep_cnt in range(ep_num):
            file_name = f"simulation_result_ep_{ep_cnt}.mat"
            if not os.path.exists(os.path.join(self.store_path, file_name)):
                print(f"Warning: {file_name} not found. Skipping...")
                continue

            mat_ep = self.load_one_ep(file_name)

            result_dic['reward'] += list(mat_ep[f"result_{ep_cnt}"]["reward"][0][0])

            for i in range(self.user_num):
                result_dic['user_capacity'][i] += list(mat_ep[f"result_{ep_cnt}"]["user_capacity"][0][0][:, i])
                result_dic['secure_capacity'][i] += list(mat_ep[f"result_{ep_cnt}"]["secure_capacity"][0][0][:, i])

            for i in range(self.attacker_num):
                result_dic['attacker_capacity'][i] += list(mat_ep[f"result_{ep_cnt}"]["attaker_capacity"][0][0][:, i])

            for i in range(self.RIS_ant_num):
                result_dic['RIS_elements'][i] += list(mat_ep[f"result_{ep_cnt}"]["reflecting_coefficient"][0][0][:, i])

        print("Data loading complete.")
        return result_dic

    def plot(self):
        """Generate and save performance plots."""
        plot_dir = os.path.join(self.store_path, 'plot', 'RIS')
        os.makedirs(plot_dir, exist_ok=True)

        print("Generating plots...")

        # Plot Reward
        plt.figure("Reward")
        plt.plot(range(len(self.all_steps['reward'])), self.all_steps['reward'], color='blue')
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Reward per Step")
        plt.savefig(os.path.join(self.store_path, 'plot', 'reward.png'))
        plt.close()

        # Plot Secure Capacity
        plt.figure("Secure Capacity")
        for i in range(self.user_num):
            plt.plot(self.all_steps['secure_capacity'][i], label=f"User {i}", color=self.color_list[i])
        plt.xlabel("Steps")
        plt.ylabel("Secure Capacity")
        plt.title("Secure Capacity per Step")
        plt.legend()
        plt.savefig(os.path.join(self.store_path, 'plot', 'secure_capacity.png'))
        plt.close()

        # Plot User Capacity
        plt.figure("User Capacity")
        for i in range(self.user_num):
            plt.plot(self.all_steps['user_capacity'][i], label=f"User {i}", color=self.color_list[i])
        plt.xlabel("Steps")
        plt.ylabel("User Capacity")
        plt.title("User Capacity per Step")
        plt.legend()
        plt.savefig(os.path.join(self.store_path, 'plot', 'user_capacity.png'))
        plt.close()

        # Plot Attacker Capacity
        plt.figure("Attacker Capacity")
        for i in range(self.attacker_num):
            plt.plot(self.all_steps['attacker_capacity'][i], label=f"Attacker {i}", color=self.color_list[i])
        plt.xlabel("Steps")
        plt.ylabel("Attacker Capacity")
        plt.title("Attacker Capacity per Step")
        plt.legend()
        plt.savefig(os.path.join(self.store_path, 'plot', 'attacker_capacity.png'))
        plt.close()

        # Plot RIS Elements
        for i in range(self.RIS_ant_num):
            self.plot_one_RIS_element(i, plot_dir)

        print("Plots saved successfully.")

    def plot_one_RIS_element(self, index, plot_dir):
        """Plot real, imaginary, and phase components of an RIS element."""
        plt.figure(figsize=(10, 5))
        ax_real_imag = plt.subplot(1, 1, 1)
        ax_phase = ax_real_imag.twinx()

        real_values = np.real(self.all_steps['RIS_elements'][index])
        imag_values = np.imag(self.all_steps['RIS_elements'][index])
        phase_values = [cmath.phase(complex_num) for complex_num in self.all_steps['RIS_elements'][index]]

        ax_real_imag.plot(real_values, color='b', label="Real")
        ax_real_imag.plot(imag_values, color='c', label="Imaginary")
        ax_phase.plot(phase_values, color='m', label="Phase", linestyle="dashed")

        ax_real_imag.set_xlabel("Steps")
        ax_real_imag.set_ylabel("Real/Imaginary")
        ax_phase.set_ylabel("Phase (radians)")
        plt.title(f"RIS Element {index}")

        ax_real_imag.legend(loc="upper left")
        ax_phase.legend(loc="upper right")

        plt.savefig(os.path.join(plot_dir, f"RIS_{index}_element.png"))
        plt.close()

    def restruct(self):
        """Save the processed data in a structured .mat file."""
        save_path = os.path.join(self.store_path, 'all_steps.mat')
        savemat(save_path, self.all_steps)
        print(f"Structured data saved at {save_path}")

if __name__ == '__main__':
    LoadPlotObject = LoadAndPlot(
        store_path="./paper/my/plot/compare/2020-12-06_15_35_34_with_RIS_16/",
        user_num=2,
        RIS_ant_num=4
    )
    LoadPlotObject.plot()
    LoadPlotObject.restruct()
