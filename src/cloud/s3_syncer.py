import os

class S3Sync:
    def sync_folder_to_s3(self,folder,aws_bucker_url):
        command=f"aws s3 sync {folder} {aws_bucker_url}"
        os.system(command)

    def sync_folder_from_s3(self,folder,aws_bucker_url):
        command=f"aws s3 sync {aws_bucker_url} {folder}"
        os.system(command)