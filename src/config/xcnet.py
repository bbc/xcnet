# XCNET - Attention-based Stylisation for Exemplar Image Colourisation
# Copyright (C) 2021  BBC Research & Development
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# General settings
input_shape = 224                       # shape of square input
epochs = 90                             # number of epochs
batch_size = 12                         # batch size
lr = 1e-5                               # learning rate

# XCNET settings
axial_attention = True                  # Use axial attention
pos_encoder = True                      # Use positional encoder
nb_scales = 2                           # Number of attention modules
hidden_dim = 256                        # Hidden dimension for all modules
output_nc = 2                           # Number of output channels

# Training loss settings
histogram_loss_coef = 2                 # Histogram loss coefficient (0 for deactivating)
pixel_loss_coef = 100                   # Pixel loss coefficient (0 for deactivating)
total_variance_loss_coef = 50           # TV loss coefficient (0 for deactivating)
gan_loss_coef = 1                       # Adversarial loss coefficient (0 for deactivating)
hist_loss_d = 0.1                       # Quantisation step for histogram loss

# Discriminator settings
disc_input_nc = 3                       # Number of input channels
disc_ndf = 64                           # Number of hidden channels
disc_norm = "batch"                     # Normalisation layer (batch, instance or none)

# Other settings
display_freq = 10                       # Display frequency of training logger
checkpoint_freq = 3000                  # Checkpoint storing frequency
seed = 42                               # Random seed
num_workers = 10                        # Number of workers