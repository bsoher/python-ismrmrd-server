# ----- 1. First stage to create a devcontainer -----
# Start from standard python-ismrmrd-server devcontainer
FROM kspacekelvin/fire-python-devcon AS python-mrd-devcontainer



# If building from the GitHub repo, uncomment the below section, open a command
# prompt in the folder containing this Dockerfile and run the command:
#    docker build --no-cache -t kspacekelvin/fire-python ./
# RUN cd /opt/code && \
#     git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git

# If doing local development, use this section to copy local code into Docker
# image. From the python-ismrmrd-server folder, uncomment the following lines
# below and run the command:
#    docker build --no-cache -t fire-python-custom -f docker/Dockerfile ./
RUN mkdir -p /opt/code/python-ismrmrd-server
COPY . /opt/code/python-ismrmrd-server


# Install any additional package dependencies here

RUN pip3 install --trusted-host pypi.org h5py==3.10.0 ismrmrd==1.14.1
RUN pip3 install --trusted-host pypi.org --no-cache-dir numpy scipy
RUN pip3 install --trusted-host pypi.org --no-cache-dir matplotlib==3.8.2 pydicom==2.4.3 pynetdicom==2.0.2


# ----- 2. Second stage to create a runtime container for deployment -----
FROM python-mrd-devcontainer AS python-mrd-runtime


# Ensure startup scripts have Unix (LF) line endings, which may not be true
# if the git repo is cloned in Windows
RUN find /opt/code/python-ismrmrd-server -name "*.sh" | xargs dos2unix

# Ensure startup scripts are marked as executable, which may be lost if files
# are copied in Windows
RUN find /opt/code/python-ismrmrd-server -name "*.sh" -exec chmod +x {} \;

# Set the starting directory so that code can use relative paths
WORKDIR /opt/code/python-ismrmrd-server

CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "--defaultConfig=invertcontrast"]

# Replace the above CMD with this ENTRYPOINT to allow allow "docker stop"
# commands to be passed to the server.  This is useful for deployments, but
# more annoying for development
# ENTRYPOINT [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]