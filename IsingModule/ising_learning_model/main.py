from IsingModule.ising_learning_model.qpu_model import QPUModel
from IsingModule.ising_learning_model.sim_anneal_model import SimAnnealModel, AnnealingSettings


def main():

    solver = "QPU"
    size = 10

    if solver == "QPU":
        print("QPU")
        profile = "default"
        # profile = "your_profile_name"
        m = QPUModel(size, profile=profile, num_reads=1)
    else:
        print("SA")
        settings_anneal = AnnealingSettings()
        settings_anneal.beta_range = [1,10]
        settings_anneal.num_reads = 100
        settings_anneal.num_sweeps = 1000
        settings_anneal.sweeps_per_beta = 1
        m = SimAnnealModel(size, settings_anneal)