from datetime import datetime

import numpy as np
from models.TrainClass import Train
from models.CGAN.model import Generator, Discriminator
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistAllLossEpoch, plotHistLossEpochGAN, plotHistPredEpochGAN
import constants.constants as constants
import torch

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('whitegrid')


class TrainModel1(Train):

    def __init__(self, gan):
        super(TrainModel1, self).__init__(gan)

    def sample_noise(self, batch_size, dim):
        return np.random.normal(0, 1, (batch_size, dim))

    def test_loss(self, G, D, testloader, criterion_loss, criterion_test_loss):
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Calculate test loss...")
            total_loss = 0
            total_loss_eye = 0
            total_loss_pose_r = 0
            total_loss_au = 0
            for iteration ,data in enumerate(testloader,0):
                input, target = data[0].to(self.device), data[1].to(self.device)
                _, target_eye, target_pose_r, target_au = self.format_data(input, target)
                input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
                target = torch.reshape(target, (-1, target.shape[2], target.shape[1]))

                real_batch_size = input.shape[0]
                noise = torch.FloatTensor(self.sample_noise(real_batch_size, constants.noise_size)).unsqueeze(1).to(self.device)

                output = G(input.float(), noise)
                output_eye, output_pose_r, output_au = self.separate_openface_features(output)
                loss_eye = criterion_test_loss(output_eye, target_eye)
                loss_pose_r = criterion_test_loss(output_pose_r, target_pose_r)
                loss_au = criterion_test_loss(output_au, target_au)
                gen_logit = D(output, input.float())
                gen_lable = torch.ones_like(gen_logit)

                adversarial_loss = criterion_loss(gen_logit, gen_lable)
                total_loss += adversarial_loss.item()
                total_loss_eye += loss_eye.item()
                total_loss_pose_r += loss_pose_r.item()
                total_loss_au += loss_au.item()

            total_loss = total_loss/(iteration + 1)
            total_loss_eye = total_loss_eye/(iteration + 1)
            total_loss_pose_r = total_loss_pose_r/(iteration + 1)
            total_loss_au = total_loss_au/(iteration + 1)
            return total_loss, total_loss_eye, total_loss_pose_r, total_loss_au


    def train_model(self):
        print("Launching of model 5 : basic GAN")
        print("Saving params...")
        G = Generator().to(self.device)
        D = Discriminator().to(self.device)

        g_opt = torch.optim.Adam(G.parameters(), lr=constants.g_lr)
        d_opt = torch.optim.Adam(D.parameters(), lr=constants.d_lr)

        save_params(constants.saved_path, G, D)
        
        bce_loss = torch.nn.BCELoss()
        criterion = torch.nn.MSELoss()


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
                _, target_eye, target_pose_r, target_au = self.format_data(inputs, targets)
                inputs = torch.reshape(inputs, (-1, inputs.shape[2], inputs.shape[1])).float()
                targets = torch.reshape(targets, (-1, targets.shape[2], targets.shape[1])).float()
                real_batch_size = inputs.shape[0]

                # * Generate fake data
                #print("** Generate fake data")
                noise = torch.FloatTensor(self.sample_noise(real_batch_size, constants.noise_size)).unsqueeze(1).to(self.device)
                fake_targets = G(inputs.float(), noise) #the generator generates the false data conditional on the prosody

                # * Train D :  maximize log(D(x)) + log(1 - D(G(z)))
                #print("** Train the discriminator")
                D.zero_grad()
                real_logit = D(targets, inputs) #produce a result for each frame (tensor of length 300)
                fake_logit = D(fake_targets.detach(), inputs)

                #discriminator prediction
                self.current_real_pred += torch.mean(real_logit).item() #moy because the discriminator made a prediction for each frame
                self.current_fake_pred += torch.mean(fake_logit).item()

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
                noise = torch.FloatTensor(self.sample_noise(real_batch_size, constants.noise_size)).unsqueeze(1).to(self.device)
                gen_y = G(inputs, noise)
                output_eye, output_pose_r, output_au = self.separate_openface_features(gen_y)
                loss_eye = criterion(output_eye, target_eye)
                loss_pose_r = criterion(output_pose_r, target_pose_r)
                loss_au = criterion(output_au, target_au)

                gen_logit = D(gen_y, inputs)
                gen_lable = torch.ones_like(gen_logit)

                g_loss = bce_loss(gen_logit, gen_lable)
                g_loss.backward()
                g_opt.step()

                if constants.unroll_steps:
                    D.load_state_dict(d_backup)

                self.current_loss_eye += loss_eye.item()
                self.current_loss_pose_r += loss_pose_r.item()
                self.current_loss_au += loss_au.item()
                self.current_loss += g_loss.item()

            self.current_loss = self.current_loss/(iteration + 1)
            self.loss_tab.append(self.current_loss)

            self.current_loss_eye = self.current_loss_eye/(iteration + 1)
            self.loss_tab_eye.append(self.current_loss_eye)

            self.current_loss_pose_r = self.current_loss_pose_r/(iteration + 1)
            self.loss_tab_pose_r.append(self.current_loss_pose_r)

            self.current_loss_au = self.current_loss_au/(iteration + 1)
            self.loss_tab_au.append(self.current_loss_au)

            #test loss
            self.t_loss, self.t_loss_eye, self.t_loss_pose_r, self.t_loss_au = self.test_loss(G, D, self.testloader, bce_loss, criterion)
            self.t_loss_tab.append(self.t_loss)
            self.t_loss_tab_eye.append(self.t_loss_eye)
            self.t_loss_tab_pose_r.append(self.t_loss_pose_r)
            self.t_loss_tab_au.append(self.t_loss_au)

            #d_loss
            self.current_d_loss = self.current_d_loss/(iteration + 1) #loss par epoch
            self.d_loss_tab.append(self.current_d_loss)
            
            #real pred
            self.current_real_pred = self.current_real_pred/(iteration + 1) 
            self.d_real_pred_tab.append(self.current_real_pred)
            
            #fake pred
            self.current_fake_pred = self.current_fake_pred/(iteration + 1)
            self.d_fake_pred_tab.append(self.current_fake_pred)

            print('[ %d ] loss : %.4f %.4f' % (epoch+1, self.current_loss, self.t_loss))
            print('[ %d ] pred : %.4f %.4f' % (epoch+1, self.current_real_pred, self.current_fake_pred))
            
            if epoch % constants.log_interval == 0 or epoch >= constants.n_epochs - 1:
                print("saving...")
                saveModel(G, epoch, constants.saved_path)
                plotHistLossEpochGAN(epoch, self.d_loss_tab, self.loss_tab, self.t_loss_tab)
                plotHistPredEpochGAN(epoch, self.d_real_pred_tab, self.d_fake_pred_tab)
                plotHistAllLossEpoch(epoch, self.loss_tab_eye, self.t_loss_tab_eye, self.loss_tab_pose_r, self.t_loss_tab_pose_r, self.loss_tab_au, self.t_loss_tab_au)

            end_epoch = datetime.now()   
            diff = end_epoch - start_epoch
            print("Duration of epoch :" + str(diff.total_seconds()))