import dataclasses
# import jax
import os

from openpi_client import websocket_client_policy

remote_url = os.environ.get('remote_url')

def parse_remote_url(remote_url: str) -> tuple[str, int]:
    try:
        remote_url = remote_url.replace('http://', '')
        host, port_str = remote_url.split(":")
        port = int(port_str)
        return host, port
    except ValueError:
        raise ValueError(f"Invalid remote_url format: '{remote_url}'. Expected format 'host:port'.")

class VLAModel():
    def __init__(self, model_name, language_only, model_type="remote", task_type="manipulation"):
    
        self.model_name = model_name
        self.model_type = model_type

        if model_type == "local":
            if "pi" in model_name:            
                config = config.get_config(model_name)
                checkpoint_dir = download.maybe_download(f"s3://openpi-assets/checkpoints/{model_name}")

                # Create a trained policy.
                self.model = policy_config.create_trained_policy(config, checkpoint_dir)
            else:
                raise ValueError(f"Model {model_name} is not supported. Please use a model from the OpenPI repository.")
            
        else:   
            if "pi" in model_name:            
                # Connect to the policy server
                remote_host, remote_port = parse_remote_url(remote_url)
                self.model = websocket_client_policy.WebsocketClientPolicy(remote_host, remote_port)
            else:
                raise ValueError(f"Model {model_name} is not supported. Please use a model from the OpenPI repository.")
            
    def respond(self, prompt, obs=None):        
        if "pi" in self.model_name:
            return self._call_openpi(prompt)
        else:
            raise ValueError(f"Model {self.model_name} is not supported. Please use a model from the OpenPI repository.")
        
    def _call_openpi(self, prompt):
        response = self.model.infer(prompt)
        pred_action = response["actions"]
        assert pred_action.shape == (10, 8)

        return pred_action

