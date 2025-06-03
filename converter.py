import torch
import torch.nn as nn
import onnx
import os
import sys
import argparse
import numpy as np
from collections import OrderedDict

# Check if TensorRT is available
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT not available. The script will convert to ONNX only.")

def inspect_pth_file(pth_path):
    """Inspect a .pth file to understand its structure"""
    print(f"Inspecting {pth_path}...")
    
    try:
        checkpoint = torch.load(pth_path, map_location='cpu')
        
        # Determine the type of checkpoint
        if isinstance(checkpoint, dict):
            print("Checkpoint is a dictionary with keys:", checkpoint.keys())
            
            # Check for common keys
            if 'model' in checkpoint:
                print("Found 'model' key")
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                print("Found 'state_dict' key")
                state_dict = checkpoint['state_dict']
            else:
                print("No standard model/state_dict key found, using checkpoint as state_dict")
                state_dict = checkpoint
        else:
            print("Checkpoint is not a dictionary, assuming it's a direct state_dict")
            state_dict = checkpoint
        
        # Analyze the state dict
        print("\nState dictionary contains the following keys:")
        for key in state_dict.keys():
            print(f"  {key}: {state_dict[key].shape}")
        
        return state_dict
    except Exception as e:
        print(f"Error inspecting .pth file: {e}")
        return None

class ContinuousActorCritic(nn.Module):
    """
    Actor-Critic network for continuous action space
    Based on the configuration provided
    """
    def __init__(self, obs_dim=10, action_dim=1):
        super(ContinuousActorCritic, self).__init__()
        
        # Define network layers based on the config
        # Note: In the actual model, this is called actor_mlp or actor_shared_mlp
        self.actor_mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        # Action mean output (mu)
        self.mu = nn.Linear(512, action_dim)
        
        # Value function (not used for inference)
        self.value = nn.Linear(512, 1)
        
        # Standard deviation for continuous actions (fixed sigma in your config)
        self.sigma = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        """Forward pass that returns all outputs for training"""
        shared_features = self.actor_mlp(obs)
        mu = self.mu(shared_features)
        value = self.value(shared_features)
        return mu, value, self.sigma.exp()
    
    def act(self, obs):
        """Forward pass for inference - returns only actions"""
        shared_features = self.actor_mlp(obs)
        mu = self.mu(shared_features)
        return mu

class PolicyInferenceWrapper(nn.Module):
    """Wrapper for exporting policy to ONNX/TensorRT"""
    def __init__(self, policy_model):
        super(PolicyInferenceWrapper, self).__init__()
        self.policy = policy_model
        
        # Input normalization stats (if available)
        self.normalize_input = True
        self.input_mean = nn.Parameter(torch.zeros(10), requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(10), requires_grad=False)
    
    def forward(self, obs):
        # Apply normalization if enabled
        if self.normalize_input:
            obs = (obs - self.input_mean) / self.input_std
        
        # Get deterministic actions
        return self.policy.act(obs)

def load_model_from_checkpoint(checkpoint_path, obs_dim=10, action_dim=1):
    """Load the model from a checkpoint file"""
    # Create model
    model = ContinuousActorCritic(obs_dim=obs_dim, action_dim=action_dim)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Process state dictionary
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Check state dict keys
    new_state_dict = {}
    
    # Map the state dict keys to our model
    for k, v in state_dict.items():
        # Skip normalization stats - we'll handle those separately
        if 'running_mean_std' in k or 'value_mean_std' in k or 'count' in k:
            continue
            
        cleaned_key = k
        
        # Handle various naming patterns
        if k.startswith('a2c_network.'):
            cleaned_key = k[len('a2c_network.'):]
        elif k.startswith('network.'):
            cleaned_key = k[len('network.'):]
        elif k.startswith('actor_critic.'):
            cleaned_key = k[len('actor_critic.'):]
        
        # Map to the correct keys in our model
        if 'actor_mlp.0' in cleaned_key or 'actor_shared_mlp.0' in cleaned_key:
            mapped_key = 'actor_mlp.0.' + cleaned_key.split('.')[-1]
        elif 'actor_mlp.2' in cleaned_key or 'actor_shared_mlp.2' in cleaned_key:
            mapped_key = 'actor_mlp.2.' + cleaned_key.split('.')[-1]
        elif 'actor_mlp.4' in cleaned_key or 'actor_shared_mlp.4' in cleaned_key:
            mapped_key = 'actor_mlp.4.' + cleaned_key.split('.')[-1]
        elif 'mu' in cleaned_key:
            mapped_key = 'mu.' + cleaned_key.split('.')[-1]
        elif 'value' in cleaned_key:
            mapped_key = 'value.' + cleaned_key.split('.')[-1]
        elif cleaned_key == 'sigma':
            mapped_key = 'sigma'
        else:
            # Skip keys we can't map
            continue
        
        new_state_dict[mapped_key] = v
    
    # Load the mapped state dict
    try:
        model.load_state_dict(new_state_dict)
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded with some missing keys (this might be fine for inference)")
    
    # Create the inference wrapper
    inference_model = PolicyInferenceWrapper(model)
    inference_model.eval()
    
    # Load input normalization stats if they exist
    if isinstance(checkpoint, dict):
        # Check different possible locations for the stats
        if 'running_mean_std' in checkpoint:
            try:
                mean = checkpoint['running_mean_std']['running_mean']
                var = checkpoint['running_mean_std']['running_var']
                with torch.no_grad():
                    inference_model.input_mean.copy_(mean)
                    inference_model.input_std.copy_(torch.sqrt(var + 1e-8))
                print("Loaded input normalization stats from checkpoint root")
            except Exception as e:
                print(f"Error loading normalization stats from checkpoint root: {e}")
        
        # Try to find stats in model dict
        elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            try:
                # Find running_mean_std keys
                mean_key = None
                var_key = None
                for k in checkpoint['model'].keys():
                    if 'running_mean_std.running_mean' in k:
                        mean_key = k
                    if 'running_mean_std.running_var' in k:
                        var_key = k
                
                if mean_key and var_key:
                    mean = checkpoint['model'][mean_key]
                    var = checkpoint['model'][var_key]
                    with torch.no_grad():
                        inference_model.input_mean.copy_(mean)
                        inference_model.input_std.copy_(torch.sqrt(var + 1e-8))
                    print("Loaded input normalization stats from model dict")
            except Exception as e:
                print(f"Error loading normalization stats from model dict: {e}")
    
    # Print a notice about normalization
    print("Input normalization is enabled - make sure to normalize observations when using this model")
    
    return inference_model

def convert_to_onnx(model, onnx_path, input_dim=10):
    """Convert PyTorch model to ONNX format"""
    print(f"Converting to ONNX: {onnx_path}")
    
    # Create dummy input
    dummy_input = torch.randn(1, input_dim)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13, 
        do_constant_folding=True,
        input_names=['input'],
        output_names=['action'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified and saved to {onnx_path}")
    
    return onnx_path

def convert_onnx_to_tensorrt(onnx_path, trt_path, precision="fp32", input_dim=10, batch_size=16):
    """Convert ONNX model to TensorRT engine"""
    if not TENSORRT_AVAILABLE:
        print("TensorRT is not available. Skipping conversion to TensorRT.")
        return None
    
    print(f"Converting ONNX to TensorRT: {trt_path}")
    
    # Initialize TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parsing error: {parser.get_error(error)}")
            return None
    
    # Configure builder
    config = builder.create_builder_config()
    
    # Set workspace size - handle API differences between TensorRT versions
    try:
        # Newer TensorRT versions
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    except AttributeError:
        try:
            # Older TensorRT versions
            config.max_workspace_size = 4 << 30  # 4GB
        except AttributeError:
            print("Warning: Could not set workspace memory limit. Continuing with default.")
    
    # Set precision mode
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("Using INT8 precision (would need calibration for accurate results)")
    else:
        print("Using FP32 precision")
    
    # Create optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, input_dim), (batch_size, input_dim), (batch_size*2, input_dim))
    config.add_optimization_profile(profile)
    
    # Build and serialize engine
    print("Building TensorRT engine (this may take a while)...")
    try:
        # Newer TensorRT versions
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build TensorRT engine")
            return None
            
        with open(trt_path, "wb") as f:
            f.write(serialized_engine)
    except AttributeError:
        try:
            # Older TensorRT versions
            engine = builder.build_engine(network, config)
            if engine is None:
                print("Failed to build TensorRT engine")
                return None
                
            with open(trt_path, "wb") as f:
                f.write(engine.serialize())
        except Exception as e:
            print(f"Failed to build TensorRT engine: {e}")
            return None
    
    print(f"TensorRT engine saved to {trt_path}")
    return trt_path

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert PyTorch RL model to TensorRT')
    parser.add_argument('--model', type=str, 
                        default=r"C:\Users\Shadow\isaaclab\runs\cartpole4_1_direct_11-10-54-35\nn\last_cartpole4_1_direct_ep_1050_rew__54313.812_.pth",
                        help='Path to the PyTorch model file')
    parser.add_argument('--obs_dim', type=int, default=10,
                        help='Observation dimension (default: 10 for quadruple cartpole)')
    parser.add_argument('--action_dim', type=int, default=1,
                        help='Action dimension (default: 1 for cartpole)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Optimal batch size for TensorRT engine')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], default='fp32',
                        help='Precision to use for TensorRT engine')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for the converted models (default: same as input)')
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract model name
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    
    # Step 1: Inspect the model file
    print("\n========== Step 1: Inspecting PyTorch model ==========")
    state_dict = inspect_pth_file(args.model)
    if state_dict is None:
        print("Failed to load model file. Exiting.")
        return
    
    # Step 2: Load the model
    print("\n========== Step 2: Loading model ==========")
    model = load_model_from_checkpoint(
        args.model,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim
    )
    
    # Step 3: Convert to ONNX
    print("\n========== Step 3: Converting to ONNX ==========")
    onnx_path = os.path.join(args.output_dir, f"{model_name}.onnx")
    convert_to_onnx(model, onnx_path, input_dim=args.obs_dim)
    
    # Step 4: Convert to TensorRT
    if TENSORRT_AVAILABLE:
        print("\n========== Step 4: Converting to TensorRT ==========")
        trt_path = os.path.join(args.output_dir, f"{model_name}.trt")
        convert_onnx_to_tensorrt(
            onnx_path, 
            trt_path, 
            precision=args.precision,
            input_dim=args.obs_dim,
            batch_size=args.batch_size
        )
    
    print("\n========== Conversion complete ==========")
    print(f"ONNX model: {onnx_path}")
    if TENSORRT_AVAILABLE:
        print(f"TensorRT engine: {trt_path}")
    else:
        print("TensorRT conversion skipped - TensorRT not available")
    print("\nTo use these models in your application, you'll need to:")
    print("1. Load the ONNX model with ONNXRuntime or")
    print("2. Load the TensorRT engine with TensorRT runtime")

if __name__ == "__main__":
    main()