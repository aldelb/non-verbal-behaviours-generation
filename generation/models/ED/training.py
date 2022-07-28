from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import constants.constants as constants
from models.TrainClass import Train
from models.ED.model import AutoEncoder
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistAllLossEpoch, plotHistLossEpoch

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('whitegrid')

class TrainModel3(Train):

    def __init__(self, gan):
        super(TrainModel3, self).__init__(gan)

    def test_loss(self, ae, testloader, criterion_pose, criterion_au):
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Calculate test loss...")
            total_loss = 0
            total_loss_eye = 0
            total_loss_pose_r = 0
            total_loss_au = 0
            for iteration, data in enumerate(testloader, 0):
                input, target = data[0].to(self.device), data[1].to(self.device)
                input, target_eye, target_pose_r, target_au = self.format_data(input, target)
                output_eye, output_pose_r, output_au = ae(input)

                loss_eye = criterion_pose(output_eye, target_eye.float())
                loss_pose_r = criterion_pose(output_pose_r, target_pose_r.float())
                loss_au = criterion_au(output_au, target_au.float())

                loss = loss_eye + loss_pose_r + loss_au
                total_loss += loss.item()
                total_loss_eye += loss_eye.item()
                total_loss_pose_r += loss_pose_r.item()
                total_loss_au += loss_au.item()

            total_loss = total_loss/(iteration + 1)
            total_loss_eye = total_loss_eye/(iteration + 1)
            total_loss_pose_r = total_loss_pose_r/(iteration + 1)
            total_loss_au = total_loss_au/(iteration + 1)
            return total_loss, total_loss_eye, total_loss_pose_r, total_loss_au


    def train_model(self):
        print("Launching of model 3 : auto encoder with four decoders")
        print("Saving params...")
        ae = AutoEncoder().to(self.device)
        optimizer = optim.Adam(ae.parameters(), lr=constants.g_lr)
        criterion = nn.MSELoss()
        save_params(constants.saved_path, ae)

        print("Starting Training Loop...")
        for epoch in range(0, self.n_epochs):
            start_epoch = datetime.now()
            self.reinitialize_loss()
            print(f"\nStarting epoch {epoch + 1}/{self.n_epochs}...")
            for iteration, data in enumerate(self.trainloader, 0):
                print("*"+f"Starting iteration {iteration + 1}/{self.n_iteration_per_epoch}...")
                torch.cuda.empty_cache()
                input, target = data[0].to(self.device), data[1].to(self.device)
                input, target_eye, target_pose_r, target_au = self.format_data(input, target)

                ae.zero_grad()

                output_eye, output_pose_r, output_au = ae(input)

                loss_eye = criterion(output_eye, target_eye)
                loss_pose_r = criterion(output_pose_r, target_pose_r)
                loss_au = criterion(output_au, target_au)

                loss = loss_eye + loss_pose_r + loss_au

                loss.backward()  # gradients are computed
                optimizer.step() # updates the parameters, the function can be called once the gradients are computed using e.g. backward().

                self.current_loss_eye += loss_eye.item()
                self.current_loss_pose_r += loss_pose_r.item()
                self.current_loss_au += loss_au.item()

                self.current_loss += loss.item()
    
            self.t_loss, self.t_loss_eye, self.t_loss_pose_r, self.t_loss_au = self.test_loss(ae, self.testloader, criterion, criterion)
            self.update_loss_tab(iteration)


            print('[ %d ] loss : %.4f %.4f' % (epoch+1, self.current_loss, self.t_loss))

            if epoch % constants.log_interval == 0 or epoch >= self.n_epochs - 1:
                print("saving...")
                plotHistLossEpoch(epoch, self.loss_tab, self.t_loss_tab)
                plotHistAllLossEpoch(epoch, self.loss_tab_eye, self.t_loss_tab_eye, self.loss_tab_pose_r, self.t_loss_tab_pose_r, self.loss_tab_au, self.t_loss_tab_au)
                saveModel(ae, epoch, constants.saved_path)

            end_epoch = datetime.now()
            diff = end_epoch - start_epoch
            print("Duration of epoch :" + str(diff.total_seconds()))
