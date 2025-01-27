import paramiko
import os
import argparse

def copy_latest_model_from_remote(remote_host, remote_folder, local_path):
    # Create an SSH connection to the remote server using the information from .ssh/config
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the .ssh/config file
        ssh_config = paramiko.SSHConfig()
        ssh_config.parse(open(os.path.expanduser('~/.ssh/config')))
        
        # Retrieve the configuration details for the remote host
        host_config = ssh_config.lookup(remote_host)
        remote_host_name = host_config.get('hostname')
        username = host_config.get('user')
        port = int(host_config.get('port', 22))  # Use port 22 by default if not specified
        key_file = host_config.get('identityfile', [None])[0]
        
        # Connect to the remote server
        if key_file:
            ssh.connect(remote_host_name, username=username, port=port, key_filename=key_file)
        else:
            ssh.connect(remote_host_name, username=username, port=port)
        
        sftp = ssh.open_sftp()

        # Check the remote directory
        try:
            remote_files = sftp.listdir(remote_folder)
        except Exception as e:
            print(f"Error accessing the remote directory: {e}")
            return
        
        # Filter files that match the pattern model_N.pt
        model_files = [f for f in remote_files if f.startswith('model_') and f.endswith('.pt')]
        model_numbers = [int(f.split('_')[1].split('.')[0]) for f in model_files]
        
        if not model_numbers:
            print(f"No 'model_N.pt' files found in {remote_folder}.")
            sftp.close()
            ssh.close()
            return
        
        # Find the file with the highest N value
        latest_model_number = max(model_numbers)
        latest_model_file = f"model_{latest_model_number}.pt"
        remote_file_path = os.path.join(remote_folder, latest_model_file)

        # Check if the file exists on the remote server
        try:
            sftp.stat(remote_file_path)  # Check if the file exists
        except FileNotFoundError:
            print(f"The file {latest_model_file} does not exist in {remote_folder}.")
            sftp.close()
            ssh.close()
            return
        
        # Create the local path structure (subdirectories)
        relative_path = os.path.relpath(remote_folder, '/home/lorebia/Github/IsaacLab/logs')
        local_full_path = os.path.join(local_path, relative_path)
        os.makedirs(local_full_path, exist_ok=True)

        # Copy the file to the local machine
        local_file_path = os.path.join(local_full_path, latest_model_file)
        sftp.get(remote_file_path, local_file_path)
        print(f"File {latest_model_file} successfully copied from {remote_folder} to {local_full_path}.")
        
        # Close the connection
        sftp.close()
        ssh.close()

    except Exception as e:
        print(f"Error during connection or file copying: {e}")

if __name__ == "__main__":
    # Set up argument parser for command line input
    parser = argparse.ArgumentParser(description="Copy the latest model file from a remote server.")
    parser.add_argument("remote_host", type=str, help="The name of the host defined in .ssh/config")
    parser.add_argument("remote_folder", type=str, help="The folder on the remote server to look for the model files")
    parser.add_argument("local_path", type=str, help="The local path where the model file will be copied")
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Call the function to copy the latest model file
    copy_latest_model_from_remote(args.remote_host, args.remote_folder, args.local_path)
