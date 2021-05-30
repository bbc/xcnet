#!/bin/sh

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

docker build -t xcnet .
mkdir -p .cache
docker run -it --mount "type=bind,src=$(pwd)/.cache,dst=/home/user/.cache/" \
--mount "type=bind,src=$(pwd)/src,dst=/app/src" --user $(id -u):$(id -g) xcnet \
python -c "from torchvision.models import vgg19; vgg19(pretrained=True, progress=False);"
