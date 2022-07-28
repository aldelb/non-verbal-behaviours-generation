from datetime import datetime
from models.TrainClass import Train
from models.AED.model import Generator, Discriminator
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistAllLossEpoch, plotHistLossEpochGAN, plotHistPredEpochGAN
import constants.constants as constants
import torch.nn as nn
import torch
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('whitegrid')


class TrainModel2(Train):

    def __init__(self, gan):
       super(TrainModel2, self).__init__(gan)

    def test_loss(self, G, D, testloader, criterion_pose, criterion_au, criterion_adv):
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Calculate test loss...")
            total_loss = 0
            total_loss_eye = 0
            total_loss_pose_r = 0
            total_loss_au = 0
            for iteration ,data in enumerate(testloader,0):
                input, target = data[0].to(self.device), data[1].to(self.device)
                input, target_eye, target_pose_r, target_au = self.format_data(input, target)

                gen_eye, gen_pose_r, gen_au = G(input)
                gen_y = torch.cat((gen_eye, gen_pose_r, gen_au), 1)
                gen_logit = D(gen_y, input)
                gen_lable = torch.ones_like(gen_logit)
                
                loss_eye = criterion_pose(gen_eye, target_eye)
                loss_pose_r = criterion_pose(gen_pose_r, target_pose_r)
                loss_au = criterion_au(gen_au, target_au)

                adversarial_loss = constants.adversarial_coeff * criterion_adv(gen_logit, gen_lable)
                
                loss = loss_eye + loss_pose_r + loss_au + adversarial_loss
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
        print("Launching of model 4 : GAN with auto encoder as generator")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        G = Generator().to(device)
        D = Discriminator().to(device)

        g_opt = torch.optim.Adam(G.parameters(), lr=constants.g_lr)
        d_opt = torch.optim.Adam(D.parameters(), lr=constants.d_lr)

        print("Saving params...")
        save_params(constants.saved_path, G, D)

        bce_loss = torch.nn.BCELoss()
        criterion = nn.MSELoss()

        print("Starting Training Loop...")
        for epoch in range(self.n_epochs):
            print(f"Starting epoch {epoch + 1}/{self.n_epochs}...")
            start_epoch = datetime.now()
            self.reinitialize_loss()
            for iteration, data in enumerate(self.trainloader, 0):
                print("*"+f"Starting iteration {iteration + 1}/{self.n_iteration_per_epoch}...")
                torch.cuda.empty_cache()
                # * Configure real data
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                inputs, target_eye, target_pose_r, target_au = self.format_data(inputs, targets)
                targets = torch.reshape(targets, (-1, targets.shape[2], targets.shape[1])).float()

                # * Generate fake data
                output_eye, output_pose_r, output_au = G(inputs) #the generator generates the false data conditional on the prosody 
                fake_targets = torch.cat((output_eye, output_pose_r, output_au), 1)

                # * Train D :  maximize log(D(x)) + log(1 - D(G(z)))
                #print("** Train the discriminator")
                D.zero_grad()
                real_logit = D(targets, inputs) #produce a result for each frame (tensor of length 300)
                fake_logit = D(fake_targets.detach(), inputs.detach())
                #discriminator prediction
                self.current_real_pred += torch.mean(real_logit).item() #moy because the discriminator made a prediction for each frame
                self.current_fake_pred += torch.mean(fake_logit).item()

                #discriminator loss
                real_label = torch.ones_like(real_logit) #tensor fill of 1 with the same size as input
                d_real_error = bce_loss(real_logit, real_label) #measures the Binary Cross Entropy between the target and the input probabilities
                d_real_error.backward()

                fake_label = torch.zeros_like(fake_logit)
                d_fake_error = bce_loss(fake_logit, fake_label)
                d_fake_error.backward()
                
                d_loss = d_real_error + d_fake_error
                d_opt.step() #updates the parameters, the function can be called once the gradients are computed using e.g. backward().

                self.current_d_loss += d_loss.item()

                if constants.unroll_steps:
                    #print("** Unroll D to reduce mode collapse")
                    # * Unroll D to reduce mode collapse
                    d_backup = D.state_dict() #a Python dictionary object that maps each layer to its parameter tensor.
                    for _ in range(constants.unroll_steps):
                        # * Train D
                        D.zero_grad()

                        real_logit = D(targets, inputs)
                        fake_logit = D(fake_targets.detach(), inputs)

                        real_label = torch.ones_like(real_logit)
                        d_real_error = bce_loss(real_logit, real_label)
                        d_real_error.backward()

                        fake_label = torch.zeros_like(fake_logit)
                        d_fake_error = bce_loss(fake_logit, fake_label)
                        d_fake_error.backward()

                        d_loss = d_real_error + d_fake_error
                        d_opt.step()

                # * Train G
                #print("** Train the generator")
                G.zero_grad()
                gen_eye, gen_pose_r, gen_au = G(inputs)
                gen_y = torch.cat((gen_eye, gen_pose_r, gen_au), 1)
                gen_logit = D(gen_y, inputs)
                gen_lable = torch.ones_like(gen_logit)

                adversarial_loss = constants.adversarial_coeff * bce_loss(gen_logit, gen_lable)
                loss_eye = criterion(gen_eye, target_eye)
                loss_pose_r = criterion(gen_pose_r, target_pose_r)
                loss_au = criterion(gen_au, target_au)

                g_loss = loss_eye + loss_pose_r + loss_au + adversarial_loss
                g_loss.backward()
                g_opt.step()

                if constants.unroll_steps:
                    D.load_state_dict(d_backup)

                self.current_loss_eye += loss_eye.item()
                self.current_loss_pose_r += loss_pose_r.item()
                self.current_loss_au += loss_au.item()

                self.current_loss += g_loss.item()

            self.t_loss, self.t_loss_eye, self.t_loss_pose_r, self.t_loss_au = self.test_loss(G, D, self.testloader, criterion, criterion, bce_loss)
            self.update_loss_tab(iteration)

            print('[ %d ] loss : %.4f %.4f' % (epoch+1, self.current_loss, self.t_loss))
            print('[ %d ] pred : %.4f %.4f' % (epoch+1, self.current_real_pred, self.current_fake_pred))
            print('adv : %.4f; loss_eye : %.4f; loss r : %.4f; loss au : %.4f' % (adversarial_loss, loss_eye, loss_pose_r, loss_au))

            if epoch % constants.log_interval == 0 or epoch >= self.n_epochs - 1:
                print("saving...")
                saveModel(G, epoch, constants.saved_path)
                plotHistLossEpochGAN(epoch, self.d_loss_tab, self.loss_tab, self.t_loss_tab)
                plotHistPredEpochGAN(epoch, self.d_real_pred_tab, self.d_fake_pred_tab)
                plotHistAllLossEpoch(epoch, self.loss_tab_eye, self.t_loss_tab_eye, self.loss_tab_pose_r, self.t_loss_tab_pose_r, self.loss_tab_au, self.t_loss_tab_au)

            end_epoch = datetime.now()   
            diff = end_epoch - start_epoch
            print("Duration of epoch :" + str(diff.total_seconds()))
