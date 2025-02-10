import paramiko
import os
import argparse

def copy_all_files_recursive(remote_host, remote_folder, local_path):
    if 'rl_games' in remote_folder:
        rl_library = 'rl_games'
    elif 'rsl_rl' in remote_folder:
        rl_library = 'rsl_rl'
    else:
        rl_library = None

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load SSH config
        ssh_config = paramiko.SSHConfig()
        ssh_config.parse(open(os.path.expanduser('~/.ssh/config')))

        # Get remote host configuration
        host_config = ssh_config.lookup(remote_host)
        remote_host_name = host_config.get('hostname')
        username = host_config.get('user')
        port = int(host_config.get('port', 22))  # Default port is 22
        key_file = host_config.get('identityfile', [None])[0]

        # Connect to the remote server
        if key_file:
            ssh.connect(remote_host_name, username=username, port=port, key_filename=key_file)
        else:
            ssh.connect(remote_host_name, username=username, port=port)

        sftp = ssh.open_sftp()

        def copy_directory(remote_dir, local_dir):
            # Create local directory if it doesn't exist
            os.makedirs(local_dir, exist_ok=True)
            print(f"Copying contents of {remote_dir} to {local_dir}")

            all_files = sftp.listdir(remote_dir)
            if rl_library == 'rsl_rl':
                files = [filename for filename in all_files if 'model' in filename]
                if not files:
                    return
                last_model = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]

            for item in all_files:
                remote_item_path = os.path.join(remote_dir, item)
                local_item_path = os.path.join(local_dir, item)

                try:
                    # Check if the item is a directory
                    if sftp.stat(remote_item_path).st_mode & 0o170000 == 0o040000:  # Directory
                        copy_directory(remote_item_path, local_item_path)
                    else:  # Regular file
                        if rl_library == 'rl_games':
                            if '_rew_' in remote_item_path:
                                continue
                            sftp.get(remote_item_path, local_item_path)
                            print(f"Copied file: {remote_item_path} to {local_item_path}")
                        elif rl_library == 'rsl_rl':
                            if 'model_' in item and item != last_model:
                                continue
                            sftp.get(remote_item_path, local_item_path)
                            print(f"Copied file: {remote_item_path} to {local_item_path}")
                        else:
                            raise NotImplementedError(f'{rl_library} not implemented yet')
                except Exception as e:
                    print(f"Error copying {item}: {e}")

        # Calculate local folder structure
        relative_path = os.path.relpath(remote_folder, '/home/lorebia/Github/IsaacLab/logs')
        local_full_path = os.path.join(local_path, relative_path)

        # Start recursive copy
        copy_directory(remote_folder, local_full_path)

        # Close connections
        sftp.close()
        ssh.close()

    except Exception as e:
        print(f"Error during connection or file copying: {e}")

if __name__ == "__main__":
    # Set up argument parser for command line input
    parser = argparse.ArgumentParser(description="Recursively copy all files and directories from a remote server directory.")
    parser.add_argument("remote_host", type=str, help="The name of the host defined in .ssh/config")
    parser.add_argument("local_path", type=str, help="The local path where the files and directories will be copied")
    parser.add_argument("remote_folder", type=str, help="The folder on the remote server to copy files from")
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Call the recursive copy function
    copy_all_files_recursive(args.remote_host, args.remote_folder, args.local_path)
