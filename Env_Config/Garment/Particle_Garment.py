import numpy as np
import torch

import omni.kit.commands
import omni.physxdemos as demo
import isaacsim.core.utils.prims as prims_utils
from pxr import Gf, UsdGeom,Sdf, UsdPhysics, PhysxSchema, UsdLux, UsdShade
from isaacsim.core.api import World
from omni.physx.scripts import physicsUtils, deformableUtils, particleUtils
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.utils.semantics import add_update_semantics, get_semantics
from isaacsim.core.utils.rotations import euler_angles_to_quat

class Particle_Garment:
    def __init__(
        self,world:World, 
        usd_path:str="Assets/Garment/Dress/Long_LongSleeve/DLLS_Dress008_0/DLLS_Dress008_0_obj.usd", 
        pos:np.ndarray=np.array([0.0, 0.0, 0.5]), 
        ori:np.ndarray=np.array([0.0, 0.0, 0.0]),
        scale:np.ndarray=np.array([0.0085, 0.0085, 0.0085]),
        visual_material_usd:str="Assets/Material/Garment/linen_Pumpkin.usd",
        particle_system_enabled:bool=True,
        enable_ccd:bool=True,
        solver_position_iteration_count:int=16,
        global_self_collision_enabled:bool=True,
        non_particle_collision_enabled:bool=False,
        contact_offset:float=0.010,             # important parameter
        rest_offset:float=0.0075,                # important parameter
        particle_contact_offset:float=0.010,    # important parameter
        fluid_rest_offset:float=0.0075,
        solid_rest_offset:float=0.0075,
        adhesion:float=0.1,                     # important parameter
        adhesion_offset_scale:float=0.0,        # important parameter
        cohesion:float=0.0,                     # important parameter
        particle_adhesion_scale:float=0.5,      # important parameter
        particle_friction_scale:float=0.5,      # important parameter
        drag:float=0.0, 
        lift:float=0.0, 
        friction:float=25.0,                    # important parameter
        damping:float=0.0,   
        gravity_scale:float=1.0,                # important parameter
        particle_mass:float=1e-2,               # important parameter
        self_collision:bool=True, 
        self_collision_filter:bool=True, 
        stretch_stiffness:float=1e12, #1e6  
        bend_stiffness:float=100.0, 
        shear_stiffness:float=100.0, 
        spring_damping:float=10.0, 
    ):
        self.world=world
        self.usd_path=usd_path
        self.position = pos
        self.orientation = ori
        self.scale = scale
        self.visual_material_usd = visual_material_usd
        self.stage=world.stage
        self.scene=world.get_physics_context()._physics_scene
        
        self.garment_view=UsdGeom.Xform.Define(self.stage,"/World/Garment")
        self.garment_name=find_unique_string_name(initial_name="garment",is_unique_fn=lambda x: not world.scene.object_exists(x))
        self.garment_prim_path=find_unique_string_name("/World/Garment/garment",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.particle_system_path=find_unique_string_name("/World/Garment/particleSystem",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.particle_material_path=find_unique_string_name("/World/Garment/particleMaterial",is_unique_fn=lambda x: not is_prim_path_valid(x))

        
        # define particle system for garment
        self.particle_system = SingleParticleSystem(
            prim_path=self.particle_system_path,
            simulation_owner=self.scene.GetPath(), 
            particle_system_enabled=particle_system_enabled, 
            enable_ccd=enable_ccd, 
            solver_position_iteration_count=solver_position_iteration_count, 
            # max_depenetration_velocity=100.0, 
            global_self_collision_enabled=global_self_collision_enabled, 
            non_particle_collision_enabled=non_particle_collision_enabled, 
            contact_offset=contact_offset,   
            rest_offset=rest_offset,      #0.010
            particle_contact_offset=particle_contact_offset,  #0.012
            fluid_rest_offset=fluid_rest_offset, 
            solid_rest_offset= solid_rest_offset,
            # wind=None, 
            # max_neighborhood=None, 
            # max_velocity=None, 
        )
        
        # define particle material for garment
        self.particle_material=ParticleMaterial(
            prim_path=self.particle_material_path, 
            adhesion=adhesion, #100.0
            adhesion_offset_scale=adhesion_offset_scale, 
            cohesion=cohesion, #100.0
            particle_adhesion_scale=particle_adhesion_scale, #1.0
            particle_friction_scale=particle_friction_scale, #1.0
            drag=drag, 
            lift=lift, 
            friction=friction, #100.0
            damping=damping,   #5.0
            gravity_scale=gravity_scale, 
            # viscosity=0.0, 
            # vorticity_confinement=0.0,
            # surface_tension=0.0, 
        )
        
        # bind particle material to particle system
        physicsUtils.add_physics_material_to_prim(
            self.stage, 
            self.stage.GetPrimAtPath(self.particle_system_path), 
            self.particle_material_path
        )


        # add garment usd to stage
        add_reference_to_stage(
            usd_path=self.usd_path,
            prim_path=self.garment_prim_path
        )
        
        # define garment Xform
        self.garment=SingleXFormPrim(
            prim_path=self.garment_prim_path,
            name=self.garment_name,
            position=self.position,
            orientation=euler_angles_to_quat(self.orientation, degrees=True),
            scale=self.scale,
        )
        
        # add particle cloth attribute to garment
        self.garment_mesh_prim_path=self.garment_prim_path+"/mesh"
        self.garment_mesh=SingleClothPrim(
            name=self.garment_name+"_mesh",
            prim_path=self.garment_mesh_prim_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
            particle_mass=particle_mass, 
            self_collision=self_collision, 
            self_collision_filter=self_collision_filter, 
            stretch_stiffness=stretch_stiffness, #1e6 
            bend_stiffness=bend_stiffness, 
            shear_stiffness=shear_stiffness, 
            spring_damping=spring_damping, 
        )
        # get particle controller
        self.particle_controller = self.garment_mesh._cloth_prim_view

        # set visual material to garemnt
        if self.visual_material_usd is not None:
            self.apply_visual_material(self.visual_material_usd)

    def set_mass(self,mass):
        physicsUtils.add_mass(self.world.stage, self.garment_mesh_prim_path, mass)
    
    def get_particle_system_id(self):
        self.particle_system_api=PhysxSchema.PhysxParticleAPI.Apply(self.particle_system.prim)
        return self.particle_system_api.GetParticleGroupAttr().Get()
    
    def apply_visual_material(self,material_path:str):
        self.visual_material_path=find_unique_string_name(self.garment_prim_path+"/visual_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
        add_reference_to_stage(usd_path=material_path,prim_path=self.visual_material_path)
        self.visual_material_prim=prims_utils.get_prim_at_path(self.visual_material_path)
        self.material_prim=prims_utils.get_prim_children(self.visual_material_prim)[0]
        self.material_prim_path=self.material_prim.GetPath()
        self.visual_material=PreviewSurface(self.material_prim_path)
        
        self.garment_mesh_prim=prims_utils.get_prim_at_path(self.garment_mesh_prim_path)
        self.garment_submesh=prims_utils.get_prim_children(self.garment_mesh_prim)
        if len(self.garment_submesh)==0:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.garment_mesh_prim_path, material_path=self.material_prim_path)
        else:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.garment_mesh_prim_path, material_path=self.material_prim_path)
            for prim in self.garment_submesh:
                omni.kit.commands.execute('BindMaterialCommand',
                prim_path=prim.GetPath(), material_path=self.material_prim_path)
        
    def get_vertice_positions(self):
        pos_world, ori_world = self.garment_mesh.get_world_pose()
        scale_world = self.garment_mesh.get_world_scale()
        mesh_points = self.garment_mesh._get_points_pose() # If backend device is torch, add '.detach().cpu().numpy()'
        transformed_mesh_points = self.transform_points(
            mesh_points,
            pos_world,     # If backend device is torch, add '.detach().cpu().numpy()'
            ori_world,     # If backend device is torch, add '.detach().cpu().numpy()'
            scale_world,   # If backend device is torch, add '.detach().cpu().numpy()'
        )
        return transformed_mesh_points
    
    def set_pose(self, pos, ori):
        if ori is not None:
            ori = euler_angles_to_quat(ori, degrees=True)
        self.garment_mesh.set_world_pose(pos, ori)
        
    def get_particle_system(self):
        return self.particle_system
        
    def get_garment_center_pos(self):
        return self.garment_mesh.get_world_pose()[0]

    def set_mass(self,mass=0.02):
        physicsUtils.add_mass(self.world.stage, self.garment_mesh_prim_path, mass)

    def transform_points(self, points, pos, ori, scale):
        """
        Transform local points to world space using position, orientation, and scale.

        Args:
            points: (N, 3) array of local points
            pos: (3,) position vector
            ori: (4,) quaternion orientation
            scale: Scale factor (numpy array)

        Returns:
            (N, 3) array of transformed points in world space
        """
        ori_matrix = self.quat_to_rot_matrix(ori)  # Expects numpy array, returns numpy array
        scaled_points = (
            points * scale
        )  # element-wise multiplication if scale is numpy array
        transformed_points = scaled_points @ ori_matrix.T + pos
        return transformed_points

    def quat_to_rot_matrix(quat: torch.Tensor) -> torch.Tensor:
        """
        Convert input quaternion to rotation matrix.
    
        Args:
            quat (torch.Tensor): (..., 4) with (w, x, y, z)
    
        Returns:
            torch.Tensor: (..., 3, 3)
        """
    
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=torch.float32)
        q = quat.clone()
        nq = torch.sum(q * q, dim=-1)
        small = nq < 1e-10
        scale = torch.sqrt(2.0 / torch.clamp(nq, min=1e-10))
        q = q * scale.unsqueeze(-1)
    
        # Outer product: (..., 4, 4)
        q_outer = q.unsqueeze(-1) @ q.unsqueeze(-2)  # (..., 4, 4)
    
        R = torch.stack(
            [
                torch.stack(
                    [
                        1.0 - q_outer[..., 2, 2] - q_outer[..., 3, 3],
                        q_outer[..., 1, 2] - q_outer[..., 3, 0],
                        q_outer[..., 1, 3] + q_outer[..., 2, 0],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        q_outer[..., 1, 2] + q_outer[..., 3, 0],
                        1.0 - q_outer[..., 1, 1] - q_outer[..., 3, 3],
                        q_outer[..., 2, 3] - q_outer[..., 1, 0],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        q_outer[..., 1, 3] - q_outer[..., 2, 0],
                        q_outer[..., 2, 3] + q_outer[..., 1, 0],
                        1.0 - q_outer[..., 1, 1] - q_outer[..., 2, 2],
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
    
        # Handle near-zero norm case
        R = torch.where(
            small.unsqueeze(-1).unsqueeze(-1),
            torch.eye(3, dtype=q.dtype, device=q.device),
            R,
        )
        return R
