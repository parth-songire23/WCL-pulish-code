import os
import numpy as np
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

    def load_all_steps(self):
        print(f"Loading data from: {self.store_path}")
        all_files = sorted([f for f in os.listdir(self.store_path) if f.startswith("simulation_result_ep_") and f.endswith(".mat")])
        ep_num = len(all_files)

        result_dic = {
            'reward': [],
            'user_capacity': [[] for _ in range(self.user_num)],
            'secure_capacity': [[] for _ in range(self.user_num)],
            'attacker_capacity': [[] for _ in range(self.attacker_num)],
            'RIS_elements': [[] for _ in range(self.RIS_ant_num)]
        }

        for ep_cnt in range(ep_num):
            file_name = f"simulation_result_ep_{ep_cnt}.mat"
            file_path = os.path.join(self.store_path, file_name)
            
            if not os.path.exists(file_path):
                print(f"Warning: {file_name} not found. Skipping...")
                continue

            mat_ep = self.load_one_ep(file_name)
            result_key = f"result_{ep_cnt}"
            if result_key not in mat_ep:
                print(f"Warning: {result_key} missing in {file_name}. Skipping...")
                continue
            
            result_data = mat_ep[result_key]

            result_dic['reward'] += list(result_data["reward"][0][0])
            
            for i in range(self.user_num):
                result_dic['user_capacity'][i] += list(result_data["user_capacity"][0][0][:, i])
                result_dic['secure_capacity'][i] += list(result_data["secure_capacity"][0][0][:, i])
            
            for i in range(self.attacker_num):
                result_dic['attacker_capacity'][i] += list(result_data["attacker_capacity"][0][0][:, i])

            for i in range(self.RIS_ant_num):
                result_dic['RIS_elements'][i] += list(result_data["reflecting_coefficient"][0][0][:, i])

        print("Data loading complete.")
        return result_dic

    def plot(self):
        """Generate and save performance plots."""
        plot_dir = os.path.join(self.store_path, 'plot', 'RIS')
        os.makedirs(plot_dir, exist_ok=True)
        print("Generating plots...")

        self._plot_metric('reward', "Reward per Step", "Reward", 'reward.png')
        self._plot_metric('secure_capacity', "Secure Capacity per Step", "Secure Capacity", 'secure_capacity.png')
        self._plot_metric('user_capacity', "User Capacity per Step", "User Capacity", 'user_capacity.png')
        self._plot_metric('attacker_capacity', "Attacker Capacity per Step", "Attacker Capacity", 'attacker_capacity.png')
        
        for i in range(self.RIS_ant_num):
            self.plot_one_RIS_element(i, plot_dir)
        
        print("Plots saved successfully.")

    def _plot_metric(self, key, title, ylabel, filename):
        """Helper function to plot different metrics."""

        if not isinstance(self.all_steps[key][0], list):
            self.all_steps[key] = [self.all_steps[key]]

        plt.figure(title)
        for i, data in enumerate(self.all_steps[key]):
            plt.plot(data, label=f"{key.capitalize()} {i}", color=self.color_list[i % len(self.color_list)])
        plt.xlabel("Steps")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(self.store_path, 'plot', filename))
        plt.close()

    def plot_one_RIS_element(self, index, plot_dir):
        """Plot real, imaginary, and phase components of an RIS element."""
        plt.figure(figsize=(10, 5))
        ax_real_imag = plt.subplot(1, 1, 1)
        ax_phase = ax_real_imag.twinx()

        ris_values = np.array(self.all_steps['RIS_elements'][index])
        real_values, imag_values = np.real(ris_values), np.imag(ris_values)
        phase_values = np.angle(ris_values)  # Using numpy's vectorized function

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
