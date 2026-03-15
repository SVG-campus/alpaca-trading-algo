import os
import subprocess
from google.cloud import compute_v1

def provision_vm(project_id, zone="us-central1-a", instance_name="alpaca-live"):
    # GCP Python Client to spin up instance
    instance_client = compute_v1.InstancesClient()
    
    # Configure machine with GPU if needed (NVIDIA T4 recommended for research)
    # This assumes the GCP_API project is configured with appropriate quotas
    print(f"Provisioning GCP VM {instance_name} in {project_id}...")
    
    # The actual API call would require defining machine_type, disk, network_interface
    # This acts as the automation entry point for the VM management
    # Updated to remove GPU as requested.
    command = f"gcloud compute instances create {instance_name} --zone={zone} --project={project_id} --machine-type=e2-standard-2 --metadata=startup-script-url=gs://your-bucket/setup_vm_service.sh"
    
    subprocess.run(command.split(), check=True)
    print("VM provisioning initiated.")

if __name__ == '__main__':
    project = os.environ.get('GCP_API_PROJECT')
    if project:
        provision_vm(project)
    else:
        print("GCP_API_PROJECT not set in .env")