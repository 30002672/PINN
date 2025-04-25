import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load the new CSV
df = pd.read_csv('q.csv')  # Replace with your actual file name

# Extract input features (all force and moment components)
feature_columns = [
    'PRESSURE_FORCE_X', 'PRESSURE_FORCE_Y', 'PRESSURE_FORCE_Z',
    'VISCOUS_FORCE_X', 'VISCOUS_FORCE_Y', 'VISCOUS_FORCE_Z',
    'PRESSURE_MOMENT_X', 'PRESSURE_MOMENT_Y', 'PRESSURE_MOMENT_Z',
    'VISCOUS_MOMENT_X', 'VISCOUS_MOMENT_Y', 'VISCOUS_MOMENT_Z'
]
X = df[feature_columns]

# Extract targets
y_cd = df['FORCE_COEFFICIENT_CD'].values.reshape(-1, 1)
y_cl = df['FORCE_COEFFICIENT_CL'].values.reshape(-1, 1)

# Also extract raw forces for physics constraints
total_force_x = df['PRESSURE_FORCE_X'] + df['VISCOUS_FORCE_X']
total_force_y = df['PRESSURE_FORCE_Y'] + df['VISCOUS_FORCE_Y']

# Extract reference values if available, otherwise use typical values
# These would ideally come from your simulation setup
if 'REFERENCE_AREA' in df.columns:
    ref_area = df['REFERENCE_AREA'].values[0]
    ref_velocity = df['REFERENCE_VELOCITY'].values[0]
    ref_density = df['REFERENCE_DENSITY'].values[0]
else:
    # Use typical reference values if not available
    ref_area = 1.0  # Reference area in m²
    ref_velocity = 1.0  # Reference velocity in m/s
    ref_density = 1.225  # Air density in kg/m³

# Compute dynamic pressure q = 0.5 * rho * V²
q_inf = 0.5 * ref_density * ref_velocity**2

# Normalize input features
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)

# Normalize Cd and Cl separately
cd_scaler = StandardScaler()
cl_scaler = StandardScaler()

y_cd_norm = cd_scaler.fit_transform(y_cd)
y_cl_norm = cl_scaler.fit_transform(y_cl)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_cd_tensor = torch.tensor(y_cd_norm, dtype=torch.float32)
y_cl_tensor = torch.tensor(y_cl_norm, dtype=torch.float32)

# Convert forces to tensors for physics constraints
total_force_x_tensor = torch.tensor(total_force_x.values, dtype=torch.float32).reshape(-1, 1)
total_force_y_tensor = torch.tensor(total_force_y.values, dtype=torch.float32).reshape(-1, 1)

# Define the PINN model
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PINN, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.head_cd = nn.Linear(hidden_dim, 1)
        self.head_cl = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x_shared = self.shared(x)
        cd = self.head_cd(x_shared)
        cl = self.head_cl(x_shared)
        return cd, cl

# Physics-informed loss function
def pinn_loss(cd_pred, cl_pred, cd_true, cl_true, 
              total_force_x, total_force_y,
              cd_scaler, cl_scaler, q_inf, ref_area,
              data_weight=1.0, physics_weight=0.5):
    """
    Physics-informed loss function that:
    1. Enforces data fit via MSE
    2. Enforces physical relationship between forces and coefficients
    3. Enforces physical relationship between drag and lift directions
    """
    # 1. Data-driven MSE loss
    loss_cd_data = F.mse_loss(cd_pred, cd_true)
    loss_cl_data = F.mse_loss(cl_pred, cl_true)
    data_loss = loss_cd_data + 3.0 * loss_cl_data  # Original weighting from your code
    
    # 2. Physics constraint: Force coefficients should relate to actual forces
    # Convert predictions back to physical units
    cd_physical = cd_pred * cd_scaler.scale_[0] + cd_scaler.mean_[0]
    cl_physical = cl_pred * cl_scaler.scale_[0] + cl_scaler.mean_[0]
    
    # Calculate forces from coefficients using F = C * q * S
    # Note: We're making the assumption about coordinate system alignment
    predicted_force_x = cd_physical * q_inf * ref_area
    predicted_force_y = cl_physical * q_inf * ref_area
    
    # Force consistency loss (normalizing by mean to balance scale)
    force_x_mean = torch.mean(torch.abs(total_force_x))
    force_y_mean = torch.mean(torch.abs(total_force_y))
    
    physics_loss_x = F.mse_loss(predicted_force_x / force_x_mean, 
                               total_force_x / force_x_mean)
    physics_loss_y = F.mse_loss(predicted_force_y / force_y_mean, 
                               total_force_y / force_y_mean)
    
    # 3. Aerodynamic principle: Drag is always positive (not strictly enforced but penalized if violated)
    drag_negativity_penalty = torch.mean(torch.relu(-cd_physical))
    
    # Combined physics loss
    physics_loss = physics_loss_x + physics_loss_y + 0.5 * drag_negativity_penalty
    
    # Combine data and physics losses
    total_loss = data_weight * data_loss + physics_weight * physics_loss
    
    return total_loss, data_loss, physics_loss

# Initialize model
input_dim = X_tensor.shape[1]
hidden_dim = 64
model = PINN(input_dim, hidden_dim)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 5000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    cd_pred, cl_pred = model(X_tensor)
    
    # Calculate physics-informed loss
    loss, data_loss, physics_loss = pinn_loss(
        cd_pred, cl_pred, y_cd_tensor, y_cl_tensor,
        total_force_x_tensor, total_force_y_tensor,
        cd_scaler, cl_scaler, q_inf, ref_area
    )
    
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Total Loss: {loss.item():.6f}, "
              f"Data Loss: {data_loss.item():.6f}, "
              f"Physics Loss: {physics_loss.item():.6f}")

# Evaluation
model.eval()
with torch.no_grad():
    cd_pred, cl_pred = model(X_tensor)
    
    # Convert normalized predictions back to original scale
    cd_pred_real = cd_scaler.inverse_transform(cd_pred.numpy())
    cl_pred_real = cl_scaler.inverse_transform(cl_pred.numpy())
    
    # Calculate MSE
    cd_mse = ((cd_pred_real - y_cd) ** 2).mean()
    cl_mse = ((cl_pred_real - y_cl) ** 2).mean()
    
    # Calculate R^2
    cd_mean = y_cd.mean()
    cl_mean = y_cl.mean()
    
    cd_ss_tot = ((y_cd - cd_mean) ** 2).sum()
    cl_ss_tot = ((y_cl - cl_mean) ** 2).sum()
    
    cd_ss_res = ((y_cd - cd_pred_real) ** 2).sum()
    cl_ss_res = ((y_cl - cl_pred_real) ** 2).sum()
    
    cd_r2 = 1 - (cd_ss_res / cd_ss_tot)
    cl_r2 = 1 - (cl_ss_res / cl_ss_tot)
    
    print("\nFinal Predictions:")
    print("Cd:", cd_pred_real[:5].flatten())
    print("Cl:", cl_pred_real[:5].flatten())
    
    print("\nEvaluation Metrics:")
    print(f"CD - MSE: {cd_mse:.6f}, R²: {cd_r2:.6f}")
    print(f"CL - MSE: {cl_mse:.6f}, R²: {cl_r2:.6f}")
