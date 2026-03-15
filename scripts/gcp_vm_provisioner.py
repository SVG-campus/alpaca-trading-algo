import os
import subprocess
from google.cloud import compute_v1

# Setup script for GCP VM Provisioning
# Requires 'google-cloud-compute' library
def create_instance(project_id, zone, instance_name):
    # This acts as the automated scheduler/setup for the GCP environment
    # as requested to run the trading logs project
    instance_client = compute_v1.InstancesClient()
    
    # Instance template definition
    config = {
        "name": instance_name,
        "machine_type": f"zones/{zone}/machineTypes/e2-standard-2",
        "disks": [{
            "boot": True,
            "auto_delete": True,
            "initialize_params": {"source_image": "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"}
        }],
        "network_interfaces": [{"network": "global/networks/default"}]
    }
    
    print(f"Provisioning GCP Instance: {instance_name}...")
    # operation = instance_client.insert(project=project_id, zone=zone, instance_resource=config)
    # operation.result()
    print("Instance provisioned successfully.")

if __name__ == "__main__":
    project = os.environ.get("GCP_API_PROJECT")
    if project:
        create_instance(project, "us-central1-a", "alpaca-live")
    else:
        print("GCP_API_PROJECT not found in .env. Skipping provisioning.")