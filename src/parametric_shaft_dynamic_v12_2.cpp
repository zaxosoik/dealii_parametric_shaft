
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector_element_access.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_postprocessor.h> 
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/index_set.h>

#include <deal.II/matrix_free/fe_evaluation_data.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>



#include <deal.II/base/symmetric_tensor.h>
 #include <deal.II/base/parameter_handler.h>

#include <deal.II/physics/transformations.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/base/hdf5.h>
#include "petscmat.h" 


#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <map>
#include <cstdlib> 

//namespace LA
//{
// #if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
//  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
//  using namespace dealii::LinearAlgebraPETSc;
//#  define USE_PETSC_LA
//#elif defined(DEAL_II_WITH_TRILINOS)
//  using namespace dealii::LinearAlgebraTrilinos;
//#else
//#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
//#endif
//} // namespace LA

using namespace dealii;
 const double pi = 3.14159265358979323846;
const double rpm_to_rps = 2*pi/60; // Conversion factor from RPM to RPS
namespace ParametricShaft
{
  
 
  namespace Parameters
  {

    struct Geometry
    {
      double point_of_origin;  
      unsigned int partitions;
      unsigned int n_global_refinements;
      bool change_diameter;
      double x_1;
      double x_2;
      double radius_final;
      bool remove_hanging_nodes;

      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);

    };
    void Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Origin",
                            "0.0",
                            Patterns::Double(0.0),
                            "x coordinate of the origin point");
        prm.declare_entry("Partitions",
                          "10",
                          Patterns::Integer(1),
                          "Number of partitions");
        prm.declare_entry("Global refinements",
                          "2",
                          Patterns::Integer(1),
                          "Number of global refinements");
        prm.declare_entry("Change Diameter",
                          "no",
                          Patterns::Selection("yes|no"),
                          "Will the diameter be changed?");
        prm.declare_entry("Diameter Change Start",
                          "0.0",
                          Patterns::Double(0.0),
                          "x coordinate of the first point of diameter change");
        prm.declare_entry("Diameter Change End",
                          "0.0",
                          Patterns::Double(0.0),
                          "x coordinate of the last point of diameter change");
        prm.declare_entry("Radius Change Value",
                          "0.35",
                          Patterns::Double(0.0),
                          "Changed radius of the shaft");
        prm.declare_entry("Remove Hanging Nodes",
                          "no",
                          Patterns::Selection("yes|no"),
                          "Will the hanging nodes be removed?");
      }
      prm.leave_subsection();
    }
    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        point_of_origin = prm.get_double("Origin");
        partitions = prm.get_integer("Partitions");
        n_global_refinements = prm.get_integer("Global refinements");
        change_diameter = prm.get_bool("Change Diameter");
        x_1 = prm.get_double("Diameter Change Start");
        x_2 = prm.get_double("Diameter Change End");
        radius_final = prm.get_double("Radius Change Value");
        remove_hanging_nodes = prm.get_bool("Remove Hanging Nodes");
        


      }
      prm.leave_subsection();
    }
    struct ShaftGeometry
    {
        
        double radius_intermediateshaft;
        double length_intermediateshaft;
        double distance_from_point_intermediateshaft;
        double rho_intermediateshaft;
        bool useShaft_intermediateshaft;
        double mass_inertia_intermediateshaft;

        double radius_engineshaft;
        double length_engineshaft;
        double distance_from_point_engineshaft;
        double rho_engineshaft;
        bool useShaft_engineshaft;
        double mass_inertia_engineshaft;

        double radius_thrustshaft;
        double length_thrustshaft;
        double distance_from_point_thrustshaft;
        double rho_thrustshaft;
        bool useShaft_thrustshaft;
        double mass_inertia_thrustshaft;

        double radius_propellershaft;
        double length_propellershaft;
        double distance_from_point_propellershaft;
        double rho_propellershaft;
        bool useShaft_propellershaft;
        double mass_inertia_propellershaft;


        static void declare_parameters(ParameterHandler &prm, const std::string subsection);
     
        void parse_parameters(ParameterHandler &prm);
    
        
    };
    void ShaftGeometry::declare_parameters(ParameterHandler &prm, const std::string subsection)
    { 
        prm.enter_subsection("ShaftGeometry");
        {
            prm.enter_subsection(subsection);
            {

                prm.declare_entry("Radius",
                                "0.5",
                                Patterns::Double(0.0),
                                "Radius of the Shaft");  
                prm.declare_entry("Length",
                                "10.0",
                                Patterns::Double(0.0),
                                "Length of the cylinder");
                prm.declare_entry("DistanceOrigin",
                                "0.0",
                                Patterns::Double(0.0),
                                "Distance from the origin");
                prm.declare_entry("Density",
                                "7700",
                                Patterns::Double(0.0),
                                "Density of the Shaft");
                prm.declare_entry("UseShaft",
                                "no",
                                Patterns::Selection("yes|no"),
                                "Will this shaft be used?");
                prm.declare_entry("MassMomentInertia",
                                "3000",
                                Patterns::Double(0.0),
                                "Mass Momente of Inertia");
            }
            prm.leave_subsection();
           
        }
        prm.leave_subsection();
    }
    void ShaftGeometry::parse_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("ShaftGeometry");
        {
            prm.enter_subsection("Intermediate");
            {
                radius_intermediateshaft = prm.get_double("Radius");
                length_intermediateshaft = prm.get_double("Length");
                distance_from_point_intermediateshaft = prm.get_double("DistanceOrigin");
                rho_intermediateshaft = prm.get_double("Density");
                useShaft_intermediateshaft = prm.get_bool("UseShaft");
                mass_inertia_intermediateshaft = prm.get_double("MassMomentInertia");
            }
            prm.leave_subsection();
            prm.enter_subsection("EngineShaft");
            {
                radius_engineshaft = prm.get_double("Radius");
                length_engineshaft = prm.get_double("Length");
                distance_from_point_engineshaft = prm.get_double("DistanceOrigin");
                rho_engineshaft = prm.get_double("Density");
                useShaft_engineshaft = prm.get_bool("UseShaft");
                mass_inertia_engineshaft = prm.get_double("MassMomentInertia");
            }
            prm.leave_subsection();
            prm.enter_subsection("ThrustShaft");
            {
                radius_thrustshaft = prm.get_double("Radius");
                length_thrustshaft = prm.get_double("Length");
                distance_from_point_thrustshaft = prm.get_double("DistanceOrigin");
                rho_thrustshaft = prm.get_double("Density");
                useShaft_thrustshaft = prm.get_bool("UseShaft");
                mass_inertia_thrustshaft = prm.get_double("MassMomentInertia");
            }
            prm.leave_subsection();
            prm.enter_subsection("PropellerShaft");
            {
                radius_propellershaft = prm.get_double("Radius");
                length_propellershaft = prm.get_double("Length");
                distance_from_point_propellershaft = prm.get_double("DistanceOrigin");
                rho_propellershaft = prm.get_double("Density");
                useShaft_propellershaft = prm.get_bool("UseShaft");
                mass_inertia_propellershaft = prm.get_double("MassMomentInertia");
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
    }

    struct ShipData
    { 
        double MCR;
        double rpm_MCR;
        double rpm;
        double v_ship;
        double prop_weight;
        double prop_length;
        double prop_mass_moment_inertia;
        double total_engine_mass_moment_inertia;
        unsigned int cylinders;
        double cylinder_mass_moment_inertia;
        double cylinder_density;
        double cylinder_length;
        bool useFlywheel;
        bool useTotalEngine;
        double flywheel_mass_moment_inertia;
        double flywheel_radius;
        double flywheel_density;
        double flywheel_length;
        double flywheel_distance_from_point;

      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
    void ShipData::declare_parameters(ParameterHandler &prm)
    {
     prm.enter_subsection("ShipData");
      {
      prm.enter_subsection("Engine");
      {
        prm.declare_entry("Use separate Cylinders",
                            "no",
                            Patterns::Selection("yes|no"),
                            "Will the engine be simulated with separate cylinders?");
        prm.declare_entry("MCR",
                          "15000.0",
                          Patterns::Double(0.0),
                          "Maximum continuous rating (kW)");
        prm.declare_entry("RPM MCR",
                          "100.0",
                          Patterns::Double(0.0),
                          "RPM at MCR");
        prm.declare_entry("RPM",
                          "100.0",
                          Patterns::Double(0.0),
                          "RPM");
        prm.declare_entry("TotalMassMomentInertia",
                            "3000.0",
                            Patterns::Double(0.0),
                            "Total Mass Moment of Inertia of the engine");
        prm.declare_entry("Cylinders",
                            "6",
                            Patterns::Integer(0),
                            "Number of cylinders");
        prm.declare_entry("CylinderMassMomentInertia",
                            "100.0",
                            Patterns::Double(0.0),
                            "Mass Moment of Inertia of the cylinder");
        prm.declare_entry("CylinderDensity",
                            "7700.0",
                            Patterns::Double(0.0),
                            "Density of the cylinder");
        prm.declare_entry("CylinderLength",
                            "1.0",
                            Patterns::Double(0.0),
                            "Length of the cylinder");
      }
         prm.leave_subsection();   
        prm.enter_subsection("Flywheel");  
        {
        prm.declare_entry("UseFlywheel",
                          "no",
                          Patterns::Selection("yes|no"),
                          "Will the flywheel be used?");
        prm.declare_entry("Flywheel Mass Moment of Inertia",
                            "100.0",
                            Patterns::Double(0.0),
                            "Mass Moment of Inertia of the flywheel");
        prm.declare_entry("Flywheel Radius",
                            "1.0",
                            Patterns::Double(0.0),
                            "Radius of the flywheel");
        prm.declare_entry("Flywheel Density",
                            "7700.0",
                            Patterns::Double(0.0),
                            "Density of the flywheel");
        prm.declare_entry("Flywheel Length",
                            "1.0",
                            Patterns::Double(0.0),
                            "Length of the flywheel");
        prm.declare_entry("Flywheel Distance from Point",
                            "1.0",
                            Patterns::Double(0.0),
                            "Distance from the origin of the flywheel");
        }
        prm.leave_subsection();
         prm.enter_subsection("Propeller");        
       {      
        
        prm.declare_entry("Propeller Weight",
                          "17",
                          Patterns::Double(0.0),
                          "Weight of the propeller (ton)");
        prm.declare_entry("Propeller Length",
                            "1.0",
                            Patterns::Double(0.0),
                            "Length of the propeller");
        prm.declare_entry("Propeller Mass Moment of Inertia",
                            "100.0",
                            Patterns::Double(0.0),
                            "Mass Moment of Inertia of the propeller");

      }
      prm.leave_subsection();
         
      prm.declare_entry("VSHIP",
                          "20.0",
                          Patterns::Double(0.0),
                          "Speed of the ship (knots)");
        }
    prm.leave_subsection();
    }
    void ShipData::parse_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("ShipData");
        {
            prm.enter_subsection("Engine");
            {
                useTotalEngine = prm.get_bool("Use separate Cylinders");
                MCR = 1000*prm.get_double("MCR");
                rpm_MCR = prm.get_double("RPM MCR");
                rpm = prm.get_double("RPM");
                total_engine_mass_moment_inertia = prm.get_double("TotalMassMomentInertia");
                cylinders = prm.get_integer("Cylinders");
                cylinder_mass_moment_inertia = prm.get_double("CylinderMassMomentInertia");
                cylinder_density = prm.get_double("CylinderDensity");
                cylinder_length = prm.get_double("CylinderLength");
            }
            prm.leave_subsection();
            prm.enter_subsection("Flywheel");
            {
                useFlywheel = prm.get_bool("UseFlywheel");
                flywheel_mass_moment_inertia = prm.get_double("Flywheel Mass Moment of Inertia");
                flywheel_radius = prm.get_double("Flywheel Radius");
                flywheel_density = prm.get_double("Flywheel Density");
                flywheel_length = prm.get_double("Flywheel Length");
                flywheel_distance_from_point = prm.get_double("Flywheel Distance from Point");
            }
            prm.leave_subsection();
            prm.enter_subsection("Propeller");
            {
                prop_length = prm.get_double("Propeller Length");
                prop_weight = prm.get_double("Propeller Weight")*1000;
                prop_mass_moment_inertia = prm.get_double("Propeller Mass Moment of Inertia");
            }
            prm.leave_subsection();
            v_ship = prm.get_double("VSHIP")*0.5144444;
            
        }
      prm.leave_subsection();
    }

    struct Bearings
    {
        unsigned int bearings_journal;
        double distance_journal;
        double length_journal;
        double radius_journal;
        double distance_between_bearings_journal;
        double eccentricity_journal;
        double phi_journal;
        double k_equiv_z_journal;
        double k_equiv_y_journal;
        double friction_coeff_journal;
        double viscocity_journal;
        double bearing_tol_journal;
        unsigned int displacement_journal_no1_ID;
        double displacement_journal_no1_displacement;
        unsigned int displacement_journal_no1_direction;
        double displacement_journal_no1_endtime;
        unsigned int displacement_journal_no2_ID;
        double displacement_journal_no2_displacement;
        unsigned int displacement_journal_no2_direction;
        double displacement_journal_no2_endtime;

        unsigned int bearings_tailtube;
        double distance_tailtube;
        double length_tailtube;
        double radius_tailtube;
        double distance_between_bearings_tailtube;
        double eccentricity_tailtube;
        double phi_tailtube;
        double k_equiv_z_tailtube;
        double k_equiv_y_tailtube;
        double friction_coeff_tailtube;
        double viscocity_tailtube;
        double bearing_tol_tailtube;
        unsigned int displacement_tailtube_no1_ID;
        double displacement_tailtube_no1_displacement;
        unsigned int displacement_tailtube_no1_direction;
        double displacement_tailtube_no1_endtime;
        unsigned int displacement_tailtube_no2_ID;
        double displacement_tailtube_no2_displacement;
        unsigned int displacement_tailtube_no2_direction;
        double displacement_tailtube_no2_endtime;

        unsigned int bearings_sterntube;
        double distance_sterntube;
        double length_sterntube;
        double radius_sterntube;
        double distance_between_bearings_sterntube;
        double eccentricity_sterntube;
        double phi_sterntube;
        double k_equiv_z_sterntube;
        double k_equiv_y_sterntube;
        double friction_coeff_sterntube;
        double viscocity_sterntube;
        double bearing_tol_sterntube;
        unsigned int displacement_sterntube_no1_ID;
        double displacement_sterntube_no1_displacement;
        unsigned int displacement_sterntube_no1_direction;
        double displacement_sterntube_no1_endtime;
        unsigned int displacement_sterntube_no2_ID;
        double displacement_sterntube_no2_displacement;
        unsigned int displacement_sterntube_no2_direction;
        double displacement_sterntube_no2_endtime;
       
        unsigned int bearings_engine;
        double distance_engine;
        double length_engine;
        double radius_engine;
        double distance_between_bearings_engine;
        double eccentricity_engine;
        double phi_engine;
        double k_equiv_z_engine;
        double k_equiv_y_engine;
        double friction_coeff_engine;
        double viscocity_engine;
        double bearing_tol_engine;
        unsigned int displacement_engine_no1_ID;
        double displacement_engine_no1_displacement;
        unsigned int displacement_engine_no1_direction;
        double displacement_engine_no1_endtime;
        unsigned int displacement_engine_no2_ID;
        double displacement_engine_no2_displacement;
        unsigned int displacement_engine_no2_direction;
        double displacement_engine_no2_endtime;




      static void declare_parameters(ParameterHandler &prm,const std::string subsection);
      void parse_parameters(ParameterHandler &prm);
    };
    void Bearings::declare_parameters(ParameterHandler &prm,const std::string subsection )
    {
      prm.enter_subsection("Bearings");
      {
        prm.enter_subsection(subsection);
        {
            prm.declare_entry("bearings",
                            "0",
                            Patterns::Selection("0|1|2|3|4|5|6|7|8|9"),
                            "Number of bearings");
            prm.enter_subsection("Geometry");
            {
            
                prm.declare_entry("Bearing Distance from Point",
                                "0.5",
                                Patterns::Double(0.0),
                                "Distance of the rearmost bearing from Origin");
                prm.declare_entry("Length of the Bearing",
                                "1.0",
                                Patterns::Double(0.0),
                                "Length of the bearing");
                prm.declare_entry("Radius Bearing",
                                "1.05",
                                Patterns::Double(0.0),
                                "Radius of the bearing");
                prm.declare_entry("Bearings distance",
                                "1.05",
                                Patterns::Double(0.0),
                                "Distance between bearings");
                prm.declare_entry ("Bearing Eccentricity",
                            "0.0001",
                            Patterns::Double(0.0),
                            "Eccentricity of Bearing");
                prm.declare_entry ("Bearing phi",
                            "3.66519",
                            Patterns::Double(0.0),
                            "Angle of max eccentricity in rad");
            }
            prm.leave_subsection();
            prm.enter_subsection("Fluid");
            {
                prm.declare_entry ("Equivalent Stiffness z",
                                    "500000",
                                    Patterns::Double(0.0),
                                    "Equivalent Stiffness of the bearing in kN/m");
                prm.declare_entry ("Equivalent Stiffness y",
                                    "500000",
                                    Patterns::Double(0.0),
                                    "Equivalent Stiffness of the bearing in kN/m");
                prm.declare_entry("Friction Coefficient",
                                "0.001",
                                Patterns::Double(0.0),
                                "Friction Coefficient of the bearing");

                prm.declare_entry("Viscocity",
                                "0.001",
                                Patterns::Double(0.0),
                                "Viscocity of the bearing");
                prm.declare_entry("Solver Tolerance",
                                "0.0001",
                                Patterns::Double(0.0),
                                "Tolerance for solving the bearing");

               
            }
            prm.leave_subsection();
            prm.enter_subsection("Displacement");
            {
                prm.enter_subsection("Bearing 1st Displacement");
                {
                    prm.declare_entry("Bearing Number",
                                    "0",
                                     Patterns::Selection("0|1|2|3|4|5|6|7|8|9"),
                                    "Number of the bearing");
                    prm.declare_entry("Bearing Displacement",
                                    "0.01",
                                    Patterns::Double(0.0),
                                    "Displacement of the bearing");
                    prm.declare_entry("Direction",
                                    "2",
                                    Patterns::Selection("1|2|3"),
                                    "Direction of the bearing Displacement");
                    prm.declare_entry("DisplacementEndtime",
                                    "1.0",
                                    Patterns::Double(0.0),
                                    "End time of the bearing Displacement");   
                }
                prm.leave_subsection();
                prm.enter_subsection("Bearing 2nd Displacement");
                {
                     prm.declare_entry("Bearing Number",
                                    "0",
                                     Patterns::Selection("0|1|2|3|4|5|6|7|8|9"),
                                    "Number of the bearing");
                    prm.declare_entry("Bearing Displacement",
                                    "0.01",
                                    Patterns::Double(0.0),
                                    "Displacement of the bearing");
                    prm.declare_entry("Direction",
                                    "2",
                                    Patterns::Selection("1|2|3"),
                                    "Direction of the displacement");
                    prm.declare_entry("DisplacementEndtime",
                                    "1.0",
                                    Patterns::Double(0.0),
                                    "End time of the displacement");   
                }
                prm.leave_subsection();
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
        }
        prm.leave_subsection();
    }
    void Bearings::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Bearings");
      {
        prm.enter_subsection("Journal");
        {
            bearings_journal = prm.get_integer("bearings");
            prm.enter_subsection("Geometry");
            {
                
                distance_journal = prm.get_double("Bearing Distance from Point");
                length_journal = prm.get_double("Length of the Bearing");
                radius_journal = prm.get_double("Radius Bearing");
                distance_between_bearings_journal = prm.get_double("Bearings distance");
                eccentricity_journal = prm.get_double("Bearing Eccentricity");
                phi_journal = prm.get_double("Bearing phi");
          
            }

            prm.leave_subsection();
            prm.enter_subsection("Fluid");
            {
                k_equiv_z_journal = prm.get_double("Equivalent Stiffness z")*1000;
                k_equiv_y_journal = prm.get_double("Equivalent Stiffness y")*1000;
                friction_coeff_journal = prm.get_double("Friction Coefficient");
                viscocity_journal = prm.get_double("Viscocity");
                bearing_tol_journal = prm.get_double("Solver Tolerance");
            }
            prm.leave_subsection();
            prm.enter_subsection("Displacement");
            {
                prm.enter_subsection("Bearing 1st Displacement");
                {
                    displacement_journal_no1_ID = prm.get_integer("Bearing Number");
                    displacement_journal_no1_displacement = prm.get_double("Bearing Displacement");
                    displacement_journal_no1_direction = prm.get_integer("Direction");
                    displacement_journal_no1_endtime = prm.get_double("DisplacementEndtime");
                }
                prm.leave_subsection();
                prm.enter_subsection("Bearing 2nd Displacement");
                {
                    displacement_journal_no2_ID = prm.get_integer("Bearing Number");
                    displacement_journal_no2_displacement = prm.get_double("Bearing Displacement");
                    displacement_journal_no2_direction = prm.get_integer("Direction");
                    displacement_journal_no2_endtime = prm.get_double("DisplacementEndtime");
                }
                prm.leave_subsection();
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
        prm.enter_subsection("TailTube");
        {
            bearings_tailtube = prm.get_integer("bearings");

            prm.enter_subsection("Geometry");
            {
            distance_tailtube = prm.get_double("Bearing Distance from Point");
            length_tailtube = prm.get_double("Length of the Bearing");
            radius_tailtube = prm.get_double("Radius Bearing");
            distance_between_bearings_tailtube = prm.get_double("Bearings distance");
            eccentricity_tailtube = prm.get_double("Bearing Eccentricity");
            phi_tailtube = prm.get_double("Bearing phi");
            }
            prm.leave_subsection();
            prm.enter_subsection("Fluid");
            {
                k_equiv_z_tailtube = prm.get_double("Equivalent Stiffness z")*1000;
                k_equiv_y_tailtube = prm.get_double("Equivalent Stiffness y")*1000;
                friction_coeff_tailtube = prm.get_double("Friction Coefficient");
                viscocity_tailtube = prm.get_double("Viscocity");
                bearing_tol_tailtube = prm.get_double("Solver Tolerance");
            }
            prm.leave_subsection();
            prm.enter_subsection("Displacement");
            {
                prm.enter_subsection("Bearing 1st Displacement");
                {
                    displacement_tailtube_no1_ID = prm.get_integer("Bearing Number");
                    displacement_tailtube_no1_displacement = prm.get_double("Bearing Displacement");
                    displacement_tailtube_no1_direction = prm.get_integer("Direction");
                    displacement_tailtube_no1_endtime = prm.get_double("DisplacementEndtime");
                }
                prm.leave_subsection();
                prm.enter_subsection("Bearing 2nd Displacement");
                {
                    displacement_tailtube_no2_ID = prm.get_integer("Bearing Number");
                    displacement_tailtube_no2_displacement = prm.get_double("Bearing Displacement");
                    displacement_tailtube_no2_direction = prm.get_integer("Direction");
                    displacement_tailtube_no2_endtime = prm.get_double("DisplacementEndtime");
                }
                prm.leave_subsection();
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
        prm.enter_subsection("SternTube");
        {
                        bearings_sterntube = prm.get_integer("bearings");

            prm.enter_subsection("Geometry");
            {
            distance_sterntube = prm.get_double("Bearing Distance from Point");
            length_sterntube = prm.get_double("Length of the Bearing");
            radius_sterntube = prm.get_double("Radius Bearing");
            distance_between_bearings_sterntube = prm.get_double("Bearings distance");
            eccentricity_sterntube = prm.get_double("Bearing Eccentricity");
            phi_sterntube = prm.get_double("Bearing phi");
            }
            prm.leave_subsection();
            prm.enter_subsection("Fluid");
            {
                k_equiv_z_sterntube = prm.get_double("Equivalent Stiffness z")*1000;
                k_equiv_y_sterntube = prm.get_double("Equivalent Stiffness y")*1000;
                friction_coeff_sterntube = prm.get_double("Friction Coefficient");
                viscocity_sterntube = prm.get_double("Viscocity");
                bearing_tol_sterntube = prm.get_double("Solver Tolerance");
            }
            prm.leave_subsection();
            prm.enter_subsection("Displacement");
            {
                prm.enter_subsection("Bearing 1st Displacement");
                {
                    displacement_sterntube_no1_ID = prm.get_integer("Bearing Number");
                    displacement_sterntube_no1_displacement = prm.get_double("Bearing Displacement");
                    displacement_sterntube_no1_direction = prm.get_integer("Direction");
                    displacement_sterntube_no1_endtime = prm.get_double("DisplacementEndtime");
                }
                prm.leave_subsection();
                prm.enter_subsection("Bearing 2nd Displacement");
                {
                    displacement_sterntube_no2_ID = prm.get_integer("Bearing Number");
                    displacement_sterntube_no2_displacement = prm.get_double("Bearing Displacement");
                    displacement_sterntube_no2_direction = prm.get_integer("Direction");
                    displacement_sterntube_no2_endtime = prm.get_double("DisplacementEndtime");
                }
                prm.leave_subsection();
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
        prm.enter_subsection("EngineBearings");
        {
                        bearings_engine = prm.get_integer("bearings");

            prm.enter_subsection("Geometry");
            {
            distance_engine = prm.get_double("Bearing Distance from Point");
            length_engine = prm.get_double("Length of the Bearing");
            radius_engine = prm.get_double("Radius Bearing");
            distance_between_bearings_engine = prm.get_double("Bearings distance");
            eccentricity_engine = prm.get_double("Bearing Eccentricity");
            phi_engine = prm.get_double("Bearing phi");
            }
            prm.leave_subsection();
            prm.enter_subsection("Fluid");
            {
                k_equiv_z_engine = prm.get_double("Equivalent Stiffness z")*1000;
                k_equiv_y_engine = prm.get_double("Equivalent Stiffness y")*1000;
                friction_coeff_engine = prm.get_double("Friction Coefficient");
                viscocity_engine = prm.get_double("Viscocity");
                bearing_tol_engine = prm.get_double("Solver Tolerance");
            }
            prm.leave_subsection();
            prm.enter_subsection("Displacement");
            {
                prm.enter_subsection("Bearing 1st Displacement");
                {
                    displacement_engine_no1_ID = prm.get_integer("Bearing Number");
                    displacement_engine_no1_displacement = prm.get_double("Bearing Displacement");
                    displacement_engine_no1_direction = prm.get_integer("Direction");
                    displacement_engine_no1_endtime = prm.get_double("DisplacementEndtime");
                }
                prm.leave_subsection();
                prm.enter_subsection("Bearing 2nd Displacement");
                {
                    displacement_engine_no2_ID = prm.get_integer("Bearing Number");
                    displacement_engine_no2_displacement = prm.get_double("Bearing Displacement");
                    displacement_engine_no2_direction = prm.get_integer("Direction");
                    displacement_engine_no2_endtime = prm.get_double("DisplacementEndtime");
                }
                prm.leave_subsection();
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
      }
        prm.leave_subsection(); 
        
    }

    struct Solver
    {
      double tol;
      unsigned int max_iter;
      unsigned int n_refinement_cycles;
      bool eigenmodes;
      unsigned int eigenmodenumber;
      std::string eigensolver;
      std::string eigenprecond;
      std::string postproscriptpath;
      double beta;
      double gamma;
      bool dynamicmode;
      std::string outputfolder;
      double HHTa;
      double NRtol;
      unsigned int NRmaxiter;
      bool useReynolds;
      bool useNR;



      static void declare_parameters(ParameterHandler &prm);
       void parse_parameters(ParameterHandler &prm);
    };

    void Solver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Solver");
      {
        prm.declare_entry("Tolerance",
                          "0.000000001",
                          Patterns::Double(0.0),
                          "Tolerance of the solver");
        prm.declare_entry("Max iterations",
                          "10000",
                          Patterns::Integer(1),
                          "Maximum number of iterations");
        prm.declare_entry("Number of refinement cycles",
                          "5",
                          Patterns::Integer(1),
                          "Number of refinement cycles");
        prm.declare_entry("Eigenmodes",
                          "yes",
                          Patterns::Selection("yes|no"),
                          "Will the eigenvalues be computed?"); 
        prm.declare_entry("EigenmodesNumber",
                          "5",
                          Patterns::Selection("5|10|All"),
                          "How many eigenmodes will be computed? For Damping Matrix All is required"); 
        prm.declare_entry("Eigensolver",
                          "KrylovSchur",
                          Patterns::Selection("KrylovSchur|GeneralizedDavidson|JacobiDavidson|Lanczos|Arnoldi"),
                          "Eigenspectrum Solver");
        prm.declare_entry("Eigenprecond",
                          "Jacobi",
                          Patterns::Selection("Jacobi|Boomer|BlockJacobi"),
                          "Eigenspectrum Preconditioner");
        prm.declare_entry("postproscriptpath",
                          "postpro_h5py_v1.py",
                          Patterns::Anything(),
                          "Path to the postprocessing script");
        prm.declare_entry("NewmarkBetaMethod Beta",
                          "0.25",
                          Patterns::Double(0.0),
                          "Beta parameter of the Newmark method");
        prm.declare_entry("NewmarkBetaMethod Gamma",
                          "0.5",
                          Patterns::Double(0.0),
                          "Gamma parameter of the Newmark method");
        prm.declare_entry("DynamicMode",
                          "yes",
                          Patterns::Selection("yes|no"),
                          "Will the dynamic mode be activated?");
        prm.declare_entry("OutputFolder",
                          "output",
                          Patterns::Anything(),
                          "Output folder");
        prm.declare_entry("NewtonTol",
                          "0.000000001",
                          Patterns::Double(0.0),
                          "Tolerance of the Newton Method");
        prm.declare_entry("NewtonMaxIter",
                          "100",
                          Patterns::Integer(1),
                          "Maximum number of iterations of the Newton Method");
        prm.declare_entry("HHTa",
                          "0.0",
                          Patterns::Double(),
                          "HHT method alpha");
        prm.declare_entry("UseReynolds",
                          "yes",
                          Patterns::Selection("yes|no"),
                          "Will the Reynolds Equation be used?");
        prm.declare_entry("UseNewton",
                          "no",
                          Patterns::Selection("yes|no"),
                          "Will the Newton Method be used?");


      }
      prm.leave_subsection();
    }
    
    void Solver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Solver");
      {
        tol = prm.get_double("Tolerance");
        max_iter = prm.get_integer("Max iterations");
        n_refinement_cycles = prm.get_integer("Number of refinement cycles");
        eigenmodes = prm.get_bool("Eigenmodes"); //(prm.parse_input_from_string("Eigenmodes")=="Yes") ? true : false;
        eigenmodenumber = prm.get_integer("EigenmodesNumber");
        eigensolver = prm.get("Eigensolver");
        eigenprecond = prm.get("Eigenprecond");
        postproscriptpath = prm.get("postproscriptpath");
        beta = prm.get_double("NewmarkBetaMethod Beta");
        gamma = prm.get_double("NewmarkBetaMethod Gamma");
        dynamicmode = prm.get_bool("DynamicMode");
        outputfolder = prm.get("OutputFolder");
        NRtol = prm.get_double("NewtonTol");
        NRmaxiter = prm.get_integer("NewtonMaxIter");
        HHTa = -prm.get_double("HHTa");
        useReynolds = prm.get_bool("UseReynolds");
        useNR = prm.get_bool("UseNewton");

        
      }
      prm.leave_subsection();
    }


    struct Materials
    {
      double youngs_modulus;
      double poisson_ratio;
      double yield_stress;
      double mu;
      double lambda;
      double rho;
      double g;
      double mu_rayleigh;
      double lambda_rayleigh;

      static void declare_parameters(ParameterHandler &prm);
       void parse_parameters(ParameterHandler &prm);
    };
    void Materials::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Materials");
      {
        prm.declare_entry("Young's Modulus",
                          "210e12",
                          Patterns::Double(0.0),
                          "Young's Modulus");
        prm.declare_entry("Poisson Ratio",
                          "0.499",
                          Patterns::Double(0.000),
                          "Poisson Ratio");
        prm.declare_entry("Yield Stress",
                          "448e9",
                          Patterns::Double(0.0),
                          "Yield Stress");
        prm.declare_entry("Mu",
                          "76.17e9",
                          Patterns::Double(0.0),
                          "Mu");
        prm.declare_entry("Lambda",
                          "96.95e9",
                          Patterns::Double(0.0),
                          "Lambda");
        prm.declare_entry("Density",
                          "7700",
                          Patterns::Double(0.0),
                          "Density");
        prm.declare_entry("Gravity",
                          "9.81",
                          Patterns::Double(0.0),
                          "Gravity");
        prm.declare_entry("Mu Rayleigh",
                          "0.025",
                          Patterns::Double(0.0),
                          "Mu Parameter for Rayleigh Damping");
        prm.declare_entry("Lambda Rayleigh",
                          "0.023",
                          Patterns::Double(0.0),
                          "Lambda Parameter for Rayleigh Damping");

      }
      prm.leave_subsection();
    
    }
    void Materials::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Materials");
      {
        youngs_modulus = prm.get_double("Young's Modulus");
        poisson_ratio = prm.get_double("Poisson Ratio");
        yield_stress = prm.get_double("Yield Stress");
        mu = prm.get_double("Mu");
        lambda = prm.get_double("Lambda");
        rho = prm.get_double("Density");
        g = prm.get_double("Gravity");
        mu_rayleigh = prm.get_double("Mu Rayleigh");
        lambda_rayleigh = prm.get_double("Lambda Rayleigh");
      }
      prm.leave_subsection();
    }

    struct Time
    {
      double delta_t;
      double end_time;
 
      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
 
    void Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");
 
        prm.declare_entry("Time step size",
                          "0.1",
                          Patterns::Double(),
                          "Time step size");
      }
      prm.leave_subsection();
    }
    void Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time = prm.get_double("End time");
        delta_t  = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }
 
 
    struct AllParameters : public Geometry,
                           public ShaftGeometry,
                           public ShipData,
                           public Bearings,
                           public Solver,
                           public Materials,
                           public Time
 
    {
      AllParameters(const std::string &input_file);
 
      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
 
    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
      //prm.print_parameters("parameters.xml");
    }
 
    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      Geometry::declare_parameters(prm);
        ShaftGeometry::declare_parameters(prm, "Intermediate");
        ShaftGeometry::declare_parameters(prm, "EngineShaft");
        ShaftGeometry::declare_parameters(prm, "ThrustShaft");
        ShaftGeometry::declare_parameters(prm, "PropellerShaft");

      ShipData::declare_parameters(prm);
      Bearings::declare_parameters(prm,"Journal");
        Bearings::declare_parameters(prm,"TailTube");
        Bearings::declare_parameters(prm,"SternTube");
        Bearings::declare_parameters(prm,"EngineBearings");
      Solver::declare_parameters(prm);
      Materials::declare_parameters(prm);
      Time::declare_parameters(prm);

    }
 
    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      Geometry::parse_parameters(prm);
      ShaftGeometry::parse_parameters(prm);
      ShipData::parse_parameters(prm);
      Bearings::parse_parameters(prm);
      Solver::parse_parameters(prm);
      Materials::parse_parameters(prm);
      Time::parse_parameters(prm);

    }
    
  
  }// namespace Parameters

  template <int dim>
  struct PointHistory
  {
    SymmetricTensor<2, dim> old_stress;
    SymmetricTensor<2,dim> old_strain;
    Point<dim> point;
    Point<dim> displacement;
  };
 
 struct Bearing {
    double start_position;
    double end_position;
    double radius_shaft;
    double radius_bearing;
    double eccentricity;
    double phi;
    double k_equiv_z;
    double k_equiv_y;
    double friction_coeff;
    double viscocity;
    double displacement_1_displacement;
    int displacement_1_direction;
    double displacement_1_endtime;
    double displacement_2_displacement;
    int displacement_2_direction;
    double displacement_2_endtime;
    int boundary_id;
};
std::vector<Bearing> combineBearings(const std::vector<Bearing>& bearings1,
                                     const std::vector<Bearing>& bearings2,
                                     const std::vector<Bearing>& bearings3,
                                     const std::vector<Bearing>& bearings4) {
    std::vector<Bearing> combinedBearings;

    combinedBearings.insert(combinedBearings.end(), bearings1.begin(), bearings1.end());
    combinedBearings.insert(combinedBearings.end(), bearings2.begin(), bearings2.end());
    combinedBearings.insert(combinedBearings.end(), bearings3.begin(), bearings3.end());
    combinedBearings.insert(combinedBearings.end(), bearings4.begin(), bearings4.end());

    return combinedBearings;
}
std::map<int, Bearing> combineBearingsMap(const std::map<int, Bearing>& bearings1,
                                       const std::map<int, Bearing>& bearings2,
                                       const std::map<int, Bearing>& bearings3,
                                       const std::map<int, Bearing>& bearings4) {
    std::map<int, Bearing> combinedBearings;

    combinedBearings.insert(bearings1.begin(), bearings1.end());
    combinedBearings.insert(bearings2.begin(), bearings2.end());
    combinedBearings.insert(bearings3.begin(), bearings3.end());
    combinedBearings.insert(bearings4.begin(), bearings4.end());

    return combinedBearings;
}
std::vector<Bearing> calculateBearings(int numBearings, 
                                        double distanceBetween, 
                                        double bearingLength, 
                                        double x_first_bearing,
                                        double radius_shaft, 
                                        double radius_bearing, 
                                        double eccentricity, 
                                        double phi, 
                                        double k_equiv_z, 
                                        double k_equiv_y, 
                                        double friction_coeff, 
                                        double viscocity, 
                                        int displacement_1_ID,
                                        double displacement_1_displacement, 
                                        int displacement_1_direction,
                                        double displacement_1_endtime,
                                        int displacement_2_ID,
                                        double displacement_2_displacement,
                                        int displacement_2_direction,
                                        double displacement_2_endtime,
                                        int boundary_id_first) {
    std::vector<Bearing> bearings;
    double currentPosition = x_first_bearing;
    int    displacement_1_ID_;
    double    displacement_1_displacement_;
    int    displacement_1_direction_ ;
    double     displacement_1_endtime_ ;
    int    displacement_2_ID_;
    double    displacement_2_displacement_;
    int    displacement_2_direction_ ;
    double     displacement_2_endtime_ ;
    for (int i = 0; i < numBearings; ++i) {
        if (displacement_1_ID==i)
        {
            displacement_1_ID_=i;
            displacement_1_displacement_ = displacement_1_displacement;
            displacement_1_direction_ = displacement_1_direction;
            displacement_1_endtime_ = displacement_1_endtime;
        }
        else
        {
            displacement_1_ID_=0;
            displacement_1_displacement_ = 0;
            displacement_1_direction_ = 0;
            displacement_1_endtime_ = 0;
        }
        if (displacement_2_ID==i)
        {
            displacement_2_ID_=i;
            displacement_2_displacement_ = displacement_2_displacement;
            displacement_2_direction_ = displacement_2_direction;
            displacement_2_endtime_ = displacement_2_endtime;
        }
        else
        {
            displacement_2_ID_=0;
            displacement_2_displacement_ = 0;
            displacement_2_direction_ = 0;
            displacement_2_endtime_ = 0;
        }
            bearings.push_back({currentPosition, 
                                currentPosition + bearingLength, 
                                radius_shaft,
                                radius_bearing, 
                                eccentricity,
                                phi,
                                k_equiv_z,
                                k_equiv_y,
                                friction_coeff,
                                viscocity,
                                displacement_1_displacement_,
                                displacement_1_direction_,
                                displacement_1_endtime_,
                                displacement_2_displacement_,
                                displacement_2_direction_,
                                displacement_2_endtime_,
                                i+boundary_id_first});        
            currentPosition += bearingLength + distanceBetween;
    }

    return bearings;
}
std::map<int, Bearing> calculateBearingsMap(int numBearings, 
                                            double distanceBetween, 
                                            double bearingLength, 
                                            double x_first_bearing,
                                            double radius_shaft, 
                                            double radius_bearing, 
                                            double eccentricity, 
                                            double phi, 
                                            double k_equiv_z, 
                                            double k_equiv_y, 
                                            double friction_coeff, 
                                            double viscocity, 
                                            int displacement_1_ID,
                                            double displacement_1_displacement, 
                                            int displacement_1_direction,
                                            double displacement_1_endtime,
                                            int displacement_2_ID,
                                            double displacement_2_displacement,
                                            int displacement_2_direction,
                                            double displacement_2_endtime,
                                            int boundary_id_first) {
    std::map<int, Bearing> bearingsMap;
    double currentPosition = x_first_bearing;

    for (int i = 0; i < numBearings; ++i) {
        int boundary_id = i + boundary_id_first;
        Bearing bearing = {
            currentPosition, 
            currentPosition + bearingLength, 
            radius_shaft,
            radius_bearing, 
            eccentricity,
            phi,
            k_equiv_z,
            k_equiv_y,
            friction_coeff,
            viscocity,
            (displacement_1_ID == i) ? displacement_1_displacement : 0,
            (displacement_1_ID == i) ? displacement_1_direction : 0,
            (displacement_1_ID == i) ? displacement_1_endtime : 0,
            (displacement_2_ID == i) ? displacement_2_displacement : 0,
            (displacement_2_ID == i) ? displacement_2_direction : 0,
            (displacement_2_ID == i) ? displacement_2_endtime : 0,
            boundary_id
        };

        bearingsMap[boundary_id] = bearing;
        currentPosition += bearingLength + distanceBetween;
    }

    return bearingsMap;
}
template <int dim>
double get_von_Mises_stress(const SymmetricTensor<2, dim> &stress)
{
    const double von_Mises_stress = std::sqrt(1.5) * (deviator(stress)).norm();

    return von_Mises_stress;
}

  template <int dim>
  //!Cijkl=tmp "STRESS-STRAIN Tensor"
  SymmetricTensor<4, dim> get_stress_strain_tensor(const double lambda,
                                                   const double mu) //For strain dependened on history we have to change this
  {
    SymmetricTensor<4, dim> tmp;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                               ((i == l) && (j == k) ? mu : 0.0) +
                               ((i == j) && (k == l) ? lambda : 0.0));
    return tmp;
  }
 
 
 
template <int dim>
inline Tensor<1, dim> get_gradient(const FEValues<dim> &fe_values,
                                  const unsigned int   shape_func,
                                  const unsigned int   q_point)
{
  Tensor<1, dim> gradient;

  for (unsigned int i = 0; i < dim; ++i)
    gradient[i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

  return gradient;
}

template <int dim>
inline Tensor<1, dim> get_gradient_face(const FEFaceValues<dim> &fe_face_values,
                                  const unsigned int       shape_func,
                                  const unsigned int       q_point)
{
  Tensor<1, dim> gradient;

  for (unsigned int i = 0; i < dim; ++i)
    gradient[i] = fe_face_values.shape_grad_component(shape_func, q_point, i)[i];

  return gradient;
}


  template <int dim>
  inline SymmetricTensor<2, dim> get_strain(const FEValues<dim> &fe_values,
                                            const unsigned int   shape_func,
                                            const unsigned int   q_point)
  {
    SymmetricTensor<2, dim> tmp;
 
    for (unsigned int i = 0; i < dim; ++i)
      tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];
 
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i + 1; j < dim; ++j)
        tmp[i][j] =
          (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
           fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
          2;
 
    return tmp;
  }
 
 
  template <int dim>
  inline SymmetricTensor<2, dim>
  get_strain(const std::vector<Tensor<1, dim>> &grad)
  {
    Assert(grad.size() == dim, ExcInternalError());
 
    SymmetricTensor<2, dim> strain;
    for (unsigned int i = 0; i < dim; ++i)
      strain[i][i] = grad[i][i];
 
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i + 1; j < dim; ++j)
        strain[i][j] = (grad[i][j] + grad[j][i]) / 2;
 
    return strain;
  }
 
 
  Tensor<2, 2> get_rotation_matrix(const std::vector<Tensor<1, 2>> &grad_u)
  {
    const double curl = (grad_u[1][0] - grad_u[0][1]);
 
    const double angle = std::atan(curl);
 
    return Physics::Transformations::Rotations::rotation_matrix_2d(-angle);
  }
 
 
  Tensor<2, 3> get_rotation_matrix(const std::vector<Tensor<1, 3>> &grad_u)
  {
    const Tensor<1, 3> curl({grad_u[2][1] - grad_u[1][2],
                             grad_u[0][2] - grad_u[2][0],
                             grad_u[1][0] - grad_u[0][1]});
 
    const double tan_angle = std::sqrt(curl * curl);
    const double angle     = std::atan(tan_angle);
 
    if (std::abs(angle) < 1e-9) //division by zero for axis
      {
        static const double rotation[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        static const Tensor<2, 3> rot(rotation);
        return rot;
      }
 
    const Tensor<1, 3> axis = curl / tan_angle;
    return Physics::Transformations::Rotations::rotation_matrix_3d(axis,
                                                                   -angle);
  }
 
 
 
 
  template <int dim>
  class TopLevel
  {
  public:
    TopLevel(const std::string &input_file);
    ~TopLevel();
    void run();
 
  private:
    //For 1st timestep ONLY
    void create_coarse_grid();
    void change_diameter(/*Triangulation<dim> &triangulation, */const double x_1, const double x_2,const double original_radius, const double radius_final);
    void refine_big_cells();
    void setup_system();
    void compute_dirichlet_constraints();
    
    // FOR ALL TIMESTEPS
    void assemble_system();
    void assemble_bearing_rhs(const Bearing bearing, const double eccentricity, const double phi);
    void interpolate_bearing_rhs();
    void predictors();
    void time_stepping(const PETScWrappers::MPI::Vector &du);
 
    void solve_timestep();

    void solve_timestep_newton();
 
    unsigned int solve_linear_problem();
    void assemble_rhs_newton(const double a_HHT);
    void newton();
    void solve_modes(std::string solver_name, std::string preconditioner_name);
   void solve_modes_serial();
    void calculate_residual_linear();
    void output_results() const;
  
    void do_initial_timestep(); //for first timestep ONLY
 
    void do_timestep();  //for ALL timesteps
 
    void refine_initial_grid(); //for first timestep ONLY
 
    
    void move_mesh(const PETScWrappers::MPI::Vector &displacement) const;   //for ALL timesteps
 
    void setup_quadrature_point_history(); // for 1st timestep to et up a pristine state for the history variables, ONLY IF the quadrature points on cells belong to the present processor
 
    void update_quadrature_point_history();


    Parameters::AllParameters prm;
    MPI_Comm mpi_communicator;
 
    parallel::shared::Triangulation<dim> triangulation;

    FESystem<dim> fe;
    //FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

 

    AffineConstraints<double> hanging_node_constraints;
    AffineConstraints<double> constraints_dirichlet_and_hanging_nodes;

    const QGauss<dim> quadrature_formula;
    const QGauss<dim-1> face_quadrature_formula;

 
    std::vector<PointHistory<dim>> quadrature_point_history; // in step_4 will use CellDataStorage, in this manually
 
 
 
    PETScWrappers::MPI::SparseMatrix system_matrix;
    PETScWrappers::MPI::SparseMatrix system_mass_matrix;
    PETScWrappers::MPI::SparseMatrix system_dynamic_matrix;
    PETScWrappers::MPI::SparseMatrix system_damping_matrix;
    PETScWrappers::MPI::SparseMatrix tmp_matrix;

    //PETScWrappers::MPI::SparseMatrix system_matrix_eigen;
    //PETScWrappers::MPI::SparseMatrix system_mass_matrix_eigen;

    //PETScWrappers::MPI::SparseMatrix cell_stiffness_matrix;
    PETScWrappers::MPI::Vector system_rhs;
    PETScWrappers::MPI::Vector system_rhs_prev;
    


    //Vector<double> displacement_n;
    //Vector<double> velocity_n;
    //Vector<double> acceleration_n;
    PETScWrappers::MPI::Vector displacement_n;
    PETScWrappers::MPI::Vector velocity_n;
    PETScWrappers::MPI::Vector acceleration_n;
    PETScWrappers::MPI::Vector acceleration_n_prev;
    PETScWrappers::MPI::Vector velocity_n_prev;
    PETScWrappers::MPI::Vector displacement_n_prev;
    PETScWrappers::MPI::Vector newton_rhs;

    std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
    std::vector<double>                eigenvalues;
    
    PETScWrappers::MPI::Vector bearing_rhs;
    PETScWrappers::MPI::Vector bearing_locally_relevant_solution;
    PETScWrappers::MPI::Vector displacement_locally_relevant_solution;
    PETScWrappers::MPI::Vector total_bearings_locally_relevant_solution;


    PETScWrappers::MPI::Vector total_displacement;

    PETScWrappers::MPI::SparseMatrix bearing_system_matrix;


    //AffineConstraints<double> constraints4modes;
    AffineConstraints<double> constraints_eigen;


   // Vector<double> locally_relevant_solution;
    //Vector<double> velocity_n;
    //PETScWrappers::MPI::Vector locally_relevant_solution;
    PETScWrappers::MPI::Vector locally_relevant_solution;
    
    double       present_time;
    double       present_timestep;
    double       end_time;
    unsigned int timestep_no;
    unsigned int newton_iter = 1;

    std::map<int, Bearing>  journal_bearings   ;
    std::map<int, Bearing>  tailtube_bearings ;
    std::map<int, Bearing>  sterntube_bearings;
    std::map<int, Bearing>  engine_bearings   ;


    // FOR PARALLEL PROCESSING
    
 
    const unsigned int n_mpi_processes;
 
    const unsigned int this_mpi_process;
 
    ConditionalOStream pcout;
 
    TimerOutput        computing_timer;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    AffineConstraints<double> constraints;
    // static const SymmetricTensor<4, dim> stress_strain_tensor;
  };
 

  //! GRAVITY, ELECTROMAGNETISM(, strong & weak interaction)
  // until line 276
  template <int dim> 
  class BodyForce : public Function<dim>
  {
  public:
    BodyForce(const double rho, const double g);
  
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  values) const override;
  
    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;

  private:
    
    const double rho;
    const double g;
  };
  
  
  template <int dim>
  BodyForce<dim>::BodyForce(const double rho,const double g)
    : Function<dim>(dim), rho(rho), g(g)
  {}
  
  
  template <int dim>
  inline void BodyForce<dim>::vector_value(const Point<dim> & /*p*/,
                                          Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);
  
    values = 0;
    values(dim - 1) = -rho * g; //z axis
  }
  
  
  template <int dim>
  void BodyForce<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list) const
  {
    const unsigned int n_points = points.size();
  
    AssertDimension(value_list.size(), n_points);
  
    for (unsigned int p = 0; p < n_points; ++p)
      BodyForce<dim>::vector_value(points[p], value_list[p]);
  }

 
 
  //! BOUNDARY CONDITIONS
  //until line 338
  template <int dim>
  class IncrementalBoundaryValues : public Function<dim>
  {
  public:
    IncrementalBoundaryValues(const double end_time,
                              const double present_timestep,
                              unsigned int direction,
                              const double displacement);
 
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  values) const override;
 
    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;
 
  private:
    const double velocity;
    const double end_time;
    const double present_timestep;
    unsigned int direction;
    const double displacement;
  };
 
 
  template <int dim>
  IncrementalBoundaryValues<dim>::IncrementalBoundaryValues(
    const double end_time,
    const double present_timestep,
    unsigned int direction,
    const double displacement)
    : Function<dim>(dim)
    , velocity(displacement/(end_time))
    , end_time(end_time)
    , present_timestep(present_timestep)
    , direction(direction)
    , displacement(displacement)
  {}
 
 
  template <int dim>
  void
  IncrementalBoundaryValues<dim>::vector_value(const Point<dim> & /*p*/,
                                               Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);
 
    values    = 0;
    //values(direction) = displacement;
    if (present_timestep >end_time)
      values(direction) = 0; //;
    else
      values(direction) = present_timestep*velocity;
    
  }
 
 
 
  template <int dim>
  void IncrementalBoundaryValues<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list) const
  {
    const unsigned int n_points = points.size();
 
    AssertDimension(value_list.size(), n_points);
 
    for (unsigned int p = 0; p < n_points; ++p)
      IncrementalBoundaryValues<dim>::vector_value(points[p], value_list[p]);
  }
 
  template <int dim>
  class DynamicIncrementalValues : public Function<dim>
  {
  public:
    DynamicIncrementalValues(const double end_time,
                              const double present_timestep,
                              unsigned int direction,
                              const double displacement);
 
    virtual void vector_value_displacement(const Point<dim> &p,
                              Vector<double> &  values_displacement) const  ;
    

 
    virtual void
    vector_value_list_displacement(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list_displacement) const ;

    virtual void vector_value_velocity(const Point<dim> &p,
                              Vector<double> &  values_velocity) const ;
    

 
    virtual void
    vector_value_list_velocity(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list_velocity) const ; 

  private:
    const double velocity;
    const double end_time;
    const double present_timestep;
    unsigned int direction;
    const double displacement;
  };
 
 
  template <int dim>
  DynamicIncrementalValues<dim>::DynamicIncrementalValues(
    const double end_time,
    const double present_timestep,
    unsigned int direction,
    const double displacement)
    : Function<dim>(dim)
    , velocity(displacement/(end_time/2))
    , end_time(end_time)
    , present_timestep(present_timestep)
    , direction(direction)
    , displacement(displacement)
  {}
 
 
  template <int dim>
  inline void DynamicIncrementalValues<dim>::vector_value_displacement(const Point<dim> & /*p*/,
                                               Vector<double> &values_displacement) const
  {
    AssertDimension(values_displacement.size(), dim);
 
    values_displacement    = 0;
    values_displacement(direction) = present_timestep * velocity;
  }
 
 
 
  template <int dim>
  inline void DynamicIncrementalValues<dim>::vector_value_list_displacement(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list_displacement) const
  {
    const unsigned int n_points = points.size();
 
    AssertDimension(value_list_displacement.size(), n_points);
 
    for (unsigned int p = 0; p < n_points; ++p)
      DynamicIncrementalValues<dim>::vector_value_displacement(points[p], value_list_displacement[p]);
  }

  template <int dim>
  inline void DynamicIncrementalValues<dim>::vector_value_velocity(const Point<dim> & /*p*/,
                                               Vector<double> &values_velocity) const
  {
    AssertDimension(values_velocity.size(), dim);
 
    values_velocity    = 0;
    values_velocity(direction) = velocity;
  }
 

  template <int dim>
  inline void DynamicIncrementalValues<dim>::vector_value_list_velocity(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list_velocity) const
  {
    const unsigned int n_points = points.size();
 
    AssertDimension(value_list_velocity.size(), n_points);
 
    for (unsigned int p = 0; p < n_points; ++p)
      DynamicIncrementalValues<dim>::vector_value_velocity(points[p], value_list_velocity[p]);
  }


/* ORIGINAL 
template <int dim>
const SymmetricTensor<4, dim> TopLevel<dim>::stress_strain_tensor =
  get_stress_strain_tensor<dim>(lambda,
                                mu);*/
    //! Calculate power to force and torque
    // Function to convert RPM to Torque
    double rpmToTorque(double rpm, double C) {
        double rps = rpm * rpm_to_rps; // Convert RPM to RPS
        double power = C * pow(rps, 3); // Calculate power using the propeller law
        double torque;
        if (rps>0)
           torque = power / ( rps); // Calculate torque
        else
           torque = 0;
        return torque;
    }
    // Function to convert RPM to Force
    double rpmToForce(double rpm, double C, double velocity) {
        double rps = rpm * rpm_to_rps; // Convert RPM to RPS
        double power = C * std::pow(rps, 3); // Calculate power using the propeller law
        
        double force;
        if (velocity<=0)
          force = 0;
        else
         force= power/velocity  ; // Calculate force
        return force;
    }
  template <int dim>
double calculate_mass_moment_of_inertia(
    const DoFHandler<dim> &dof_handler,
    const Function<dim> &density_function,
    const double x0,
    const double x1,
    const Quadrature<dim> &quadrature_formula)
{
    double total_mass_moment_of_inertia = 0.0;

   // QGauss<dim> quadrature_formula(1);
    FEValues<dim> fe_values(dof_handler.get_fe(),quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
    std::vector<double> density_values(quadrature_formula.size());

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

    for (; cell != endc; ++cell)
    {
        fe_values.reinit(cell);
        density_function.value_list(fe_values.get_quadrature_points(), density_values);

        for (unsigned int q_index = 0; q_index <quadrature_formula.size(); ++q_index)
        {
            const Point<dim> &point = fe_values.quadrature_point(q_index);
            if (point(0) >= x0 && point(0) <= x1)
            {
                double density = density_values[q_index];
                double volume_element = fe_values.JxW(q_index);
                double mass_element = density * volume_element;

                // Assuming mass is concentrated at the centroid of the cell
                Point<dim> centroid = cell->center();
                double distance_to_axis = centroid.distance(Point<dim>(point(0), 0, 0)); // Adjust for other dimensions
                total_mass_moment_of_inertia += mass_element * distance_to_axis * distance_to_axis;
            }
        }
    }

    return total_mass_moment_of_inertia;
}
double calculate_cylinder_radius(const double length, const double mass_moment_of_inertia, const double density) {
    if (density <= 0 || length <= 0 || mass_moment_of_inertia <= 0) {
        throw std::invalid_argument("Invalid input: density, length, and mass moment of inertia must be positive.");
    }

    double radius = std::sqrt((2 * mass_moment_of_inertia) / (density * M_PI * length));
    return radius;
}
double calculate_cylinder_density(const double length, const double mass_moment_of_inertia, const double radius) {
    if (radius <= 0 || length <= 0 || mass_moment_of_inertia <= 0) {
        throw std::invalid_argument("Invalid input: radius, length, and mass moment of inertia must be positive.");
    }

    double density = (2 * mass_moment_of_inertia) / (M_PI * radius * radius * length);
    return density;
}

  template<int dim>
  class Force4Area : public Function<dim>
  {
    public:
      Force4Area(const double pressure, unsigned int direction );

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override;
      virtual void 
      vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;
      
    private:
      const double pressure;  
      const unsigned int direction;
  };

  template <int dim>
  Force4Area<dim>::Force4Area(const double pressure, unsigned int direction )
    : Function<dim>(dim), pressure(pressure), direction(direction)
  {}

  template <int dim>
  inline void Force4Area<dim>::vector_value(const Point<dim> &/*p*/,
                                                Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);
    values = 0;
    values(direction) = pressure;

  }

  template <int dim>
  inline void Force4Area<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list) const
  {
    const unsigned int n_points = points.size();
 
    AssertDimension(value_list.size(), n_points);
 
    for (unsigned int p = 0; p < n_points; ++p)
      Force4Area<dim>::vector_value(points[p], value_list[p]);
  }
  template<int dim>
class springbearingLHS : public Function<dim>
{
public:
    springbearingLHS(const double k_per_area,unsigned int direction);

    virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
    virtual void 
    vector_value_list(const std::vector<Point<dim>> &points,
                    std::vector<Vector<double>> &  value_list) const override;
    
private:
    const double k_per_area; 
    unsigned int direction;
 
    
};

template <int dim>
springbearingLHS<dim>::springbearingLHS(const double k_per_area, unsigned int direction)
: Function<dim>(dim), k_per_area(k_per_area), direction(direction)
{}

template <int dim>
void springbearingLHS<dim>::vector_value(const Point<dim> &p,
                                            Vector<double> &values) const
{
AssertDimension(values.size(), dim);


values = 0;
//values(direction) = k_per_area*sin(angle2direction);
values(direction) = k_per_area;

}

template <int dim>
 void springbearingLHS<dim>::vector_value_list(
const std::vector<Point<dim>> &points,
std::vector<Vector<double>> &  value_list) const
{
const unsigned int n_points = points.size();

AssertDimension(value_list.size(), n_points);

for (unsigned int p = 0; p < n_points; ++p)
    springbearingLHS<dim>::vector_value(points[p], value_list[p]);
}
  template<int dim>
class springbearingRHS : public Function<dim>
{
public:
    springbearingRHS(const double k_per_area,unsigned int direction);

    virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
    virtual void 
    vector_value_list(const std::vector<Point<dim>> &points,
                    std::vector<Vector<double>> &  value_list) const override;
    
private:
    const double k_per_area; 
    unsigned int direction;
 
    
};

template <int dim>
springbearingRHS<dim>::springbearingRHS(const double k_per_area, unsigned int direction)
: Function<dim>(dim), k_per_area(k_per_area), direction(direction)
{}

template <int dim>
void springbearingRHS<dim>::vector_value(const Point<dim> &p,
                                            Vector<double> &values) const
{
AssertDimension(values.size(), dim);


values = 0;
//values(direction) = k_per_area*sin(angle2direction);
values(direction) = k_per_area;

}

template <int dim>
 void springbearingRHS<dim>::vector_value_list(
const std::vector<Point<dim>> &points,
std::vector<Vector<double>> &  value_list) const
{
const unsigned int n_points = points.size();

AssertDimension(value_list.size(), n_points);

for (unsigned int p = 0; p < n_points; ++p)
    springbearingRHS<dim>::vector_value(points[p], value_list[p]);
}
template <int dim>
double calculate_boundary_area(const DoFHandler<dim> &dof_handler, const unsigned int boundary_id) {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    double total_area = 0.0;

    FEFaceValues<dim> fe_face_values(dof_handler.get_fe(),
                                     QGauss<dim - 1>(dof_handler.get_fe().degree + 1),
                                     update_JxW_values | update_quadrature_points);

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

    for (; cell != endc; ++cell) {
        if (cell->at_boundary()) {
            for (unsigned int face = 0; face < n_faces_per_cell; ++face) {
                if (cell->face(face)->boundary_id() == boundary_id) {
                    fe_face_values.reinit(cell, face);
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
                        total_area += fe_face_values.JxW(q);
                    }
                }
            }
        }
    }

    return total_area;
}
template <int dim>
std::pair<double, Point<dim>> calculate_boundary_area_and_centroid(const DoFHandler<dim> &dof_handler, const unsigned int boundary_id) {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    double total_area = 0.0;
    Point<dim> centroid;

    FEFaceValues<dim> fe_face_values(dof_handler.get_fe(),
                                     QGauss<dim - 1>(dof_handler.get_fe().degree + 1),
                                     update_JxW_values | update_quadrature_points);

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

    for (; cell != endc; ++cell) {
        if (cell->at_boundary()) {
            for (unsigned int face = 0; face < n_faces_per_cell; ++face) {
                if (cell->face(face)->boundary_id() == boundary_id) {
                    fe_face_values.reinit(cell, face);
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
                        double JxW = fe_face_values.JxW(q);
                        total_area += JxW;
                        centroid += JxW * fe_face_values.quadrature_point(q);
                    }
                }
            }
        }
    }

    // Normalize the centroid by the total area to get the average position
    if (total_area > std::numeric_limits<double>::epsilon()) {
        centroid /= total_area;
    }

    return std::make_pair(total_area, centroid);
}
int ceilDiv(int numerator, int denominator) {
    if (denominator == 0) {
        throw std::runtime_error("Division by zero");
    }

    return (numerator + denominator - 1) / denominator;
}
template <int dim>
bool point_inside_box(const Point<dim> &point, const Point<dim> &min_coordinate, const Point<dim> &max_coordinate) {
    for (unsigned int i = 0; i < dim; ++i) {
        if (point[i] < min_coordinate[i] || point[i] > max_coordinate[i]) {
            return false;
        }
    }
    return true;
}

template <int dim>
std::tuple<double, Point<dim>, double> calculate_boundary_area_and_centroid(
    const DoFHandler<dim> &dof_handler,
    const unsigned int boundary_id,
    const Point<dim> &min_coordinate,
    const Point<dim> &max_coordinate,
    bool calculate_polar_moment) {
    
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    double total_area = 0.0;
    Point<dim> centroid;
    double polar_moment_of_inertia = 0.0;

    FEFaceValues<dim> fe_face_values(dof_handler.get_fe(),
                                     QGauss<dim - 1>(dof_handler.get_fe().degree + 1),
                                     update_JxW_values | update_quadrature_points);

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

    for (; cell != endc; ++cell) {
        if (cell->at_boundary()) {
            for (unsigned int face = 0; face < n_faces_per_cell; ++face) {
                if (cell->face(face)->boundary_id() == boundary_id) {
                    fe_face_values.reinit(cell, face);
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
                        Point<dim> point = fe_face_values.quadrature_point(q);
                        // Check if the point is within the specified coordinate range
                        if (point_inside_box(point, min_coordinate, max_coordinate)) {
                            double JxW = fe_face_values.JxW(q);
                            total_area += JxW;
                            centroid += JxW * point;
                            if (calculate_polar_moment) {
                                double r = point.distance(centroid); // Distance from the centroid
                                polar_moment_of_inertia += JxW * r * r; // r^2 * area element
                            }
                        }
                    }
                }
            }
        }
    }

    // Normalize the centroid by the total area to get the average position
    if (total_area > std::numeric_limits<double>::epsilon()) {
        centroid /= total_area;
    }

    return std::make_tuple(total_area, centroid, polar_moment_of_inertia);
}
template <int dim>
std::pair<double, Point<dim>> calculate_boundary_area_and_centroid(
    const DoFHandler<dim> &dof_handler,
    const unsigned int boundary_id,
    const Point<dim> &min_coordinate,
    const Point<dim> &max_coordinate) {
    
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    double total_area = 0.0;
    Point<dim> centroid;
    //double polar_moment_of_inertia = 0.0;

    FEFaceValues<dim> fe_face_values(dof_handler.get_fe(),
                                     QGauss<dim - 1>(dof_handler.get_fe().degree + 1),
                                     update_JxW_values | update_quadrature_points);

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

    for (; cell != endc; ++cell) {
        if (cell->at_boundary()) {
            for (unsigned int face = 0; face < n_faces_per_cell; ++face) {
                if (cell->face(face)->boundary_id() == boundary_id) {
                    fe_face_values.reinit(cell, face);
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
                        Point<dim> point = fe_face_values.quadrature_point(q);
                        // Check if the point is within the specified coordinate range
                        if (point_inside_box(point, min_coordinate, max_coordinate)) {
                            double JxW = fe_face_values.JxW(q);
                            total_area += JxW;
                            centroid += JxW * point;
                            //if (calculate_polar_moment) {
                             //   double r = point.distance(centroid); // Distance from the centroid
                                //polar_moment_of_inertia += JxW * r * r; // r^2 * area element
                            
                        }
                    }
                }
            }
        }
    }

    // Normalize the centroid by the total area to get the average position
    if (total_area > std::numeric_limits<double>::epsilon()) {
        centroid /= total_area;
    }

    return std::make_pair(total_area, centroid);
}
// Helper function to check if a point is inside a box defined by min and max coordinates

template <int dim>
std::pair<double, Point<dim>> calculate_boundary_area_and_centroid_with_constraint(const DoFHandler<dim> &dof_handler, const unsigned int boundary_id, const int direction, const double distance_from_0_min, const double distance_from_0_max) {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    double total_area = 0.0;
    Point<dim> centroid;

    FEFaceValues<dim> fe_face_values(dof_handler.get_fe(),
                                     QGauss<dim - 1>(dof_handler.get_fe().degree + 1),
                                     update_JxW_values | update_quadrature_points);

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

    for (; cell != endc; ++cell) {
        if (cell->at_boundary()) {
            for (unsigned int face = 0; face < n_faces_per_cell; ++face) {
                if (cell->face(face)->boundary_id() == boundary_id) {
                    fe_face_values.reinit(cell, face);
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
                        Point<dim> point = fe_face_values.quadrature_point(q);
                        
                            if(point(direction)<=distance_from_0_max && point(direction)>=distance_from_0_min )
                            {   
                                double JxW = fe_face_values.JxW(q);
                                total_area += JxW;
                                centroid += JxW * fe_face_values.quadrature_point(q);
                            }
                       
                    }
                }
            }
        }
    }

    // Normalize the centroid by the total area to get the average position
    if (total_area > std::numeric_limits<double>::epsilon()) {
        centroid /= total_area;
    }

    return std::make_pair(total_area, centroid);
}
template <int dim>
std::tuple<double, Point<dim>, double> calculate_boundary_properties(const DoFHandler<dim> &dof_handler, const unsigned int boundary_id, bool calculate_polar_moment) {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    double total_area = 0.0;
    Point<dim> centroid;
    double polar_moment_of_inertia = 0.0;

    FEFaceValues<dim> fe_face_values(dof_handler.get_fe(),
                                     QGauss<dim - 1>(dof_handler.get_fe().degree + 1),
                                     update_JxW_values | update_quadrature_points);

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

    for (; cell != endc; ++cell) {
        if (cell->at_boundary()) {
            for (unsigned int face = 0; face < n_faces_per_cell; ++face) {
                if (cell->face(face)->boundary_id() == boundary_id) {
                    fe_face_values.reinit(cell, face);
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
                        double JxW = fe_face_values.JxW(q);
                        total_area += JxW;
                        centroid += JxW * fe_face_values.quadrature_point(q);
                    }
                }
            }
        }
    }

    // Normalize the centroid by the total area to get the average position
    if (total_area > std::numeric_limits<double>::epsilon()) {
        centroid /= total_area;
    }

    // Calculate the polar moment of inertia if requested
    if (calculate_polar_moment) {
        for (cell = dof_handler.begin_active(); cell != endc; ++cell) {
            if (cell->at_boundary()) {
                for (unsigned int face = 0; face < n_faces_per_cell; ++face) {
                    if (cell->face(face)->boundary_id() == boundary_id) {
                        fe_face_values.reinit(cell, face);
                        for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
                            double JxW = fe_face_values.JxW(q);
                            Point<dim> point = fe_face_values.quadrature_point(q);
                            double r = point.distance(centroid); // Distance from the centroid
                            polar_moment_of_inertia += JxW * r * r; // r^2 * area element
                        }
                    }
                }
            }
        }
    }

    return std::make_tuple(total_area, centroid, polar_moment_of_inertia);
}
template <int dim>
void calculate_and_store_boundary_areas_and_centroids(
    const DoFHandler<dim> &dof_handler,
    const Point<dim> &min_coordinate,
    const Point<dim> &max_coordinate,
    bool print,
    std::map<types::boundary_id, std::pair<double, Point<dim>>> &boundary_info) {
    
    // This map will store the area and centroid for each boundary ID
   // std::map<unsigned int, std::tuple<double, Point<dim>, double>> boundary_info;

    // Iterate over all faces and collect unique boundary IDs greater than 3
    std::set<unsigned int> boundary_ids;
    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
    for (; cell != endc; ++cell) {
        if (cell->at_boundary()) {
            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    unsigned int boundary_id = cell->face(face)->boundary_id();
                    if (boundary_id > 3) {
                        boundary_ids.insert(boundary_id);
                    }
                }
            }
        }
    }

    // Calculate area and centroid for each boundary ID
    for (unsigned int boundary_id : boundary_ids) {
        auto [area, centroid, polar_moment] = calculate_boundary_area_and_centroid(
            dof_handler, boundary_id, min_coordinate, max_coordinate, false);
        boundary_info[boundary_id] = std::make_pair(area, centroid); //, polar_moment);
    }

    // Output the results
    if (print)
      for (const auto &boundary : boundary_info) {
          std::cout << "Boundary ID: " << boundary.first << std::endl;
          std::cout << "Area: " << std::get<0>(boundary.second) << std::endl;
          std::cout << "Centroid: " << std::get<1>(boundary.second) << std::endl;
          
          std::cout << std::endl;
      }
}

template <int dim>
std::map<types::boundary_id, std::pair<double, double>> getAllBearingSpringK(const std::map<int, Bearing>& bearings, const DoFHandler<dim> &dof_handler) {
    std::map<types::boundary_id, std::pair<double, double>> boundaryIdToValuesMap;

    for (const auto& bearingEntry : bearings) {
        const auto& bearing = bearingEntry.second; // Access the Bearing object

        const Point<dim> max_coordinate_z(bearing.start_position, bearing.radius_shaft, -bearing.radius_shaft * 0.8);
        const Point<dim> min_coordinate_z(bearing.end_position, -bearing.radius_shaft, -bearing.radius_shaft);
        const Point<dim> max_coordinate_y(bearing.start_position, bearing.radius_shaft * 0.8, bearing.radius_shaft);
        const Point<dim> min_coordinate_y(bearing.end_position, bearing.radius_shaft, -bearing.radius_shaft);
        
        auto result_1 = calculate_boundary_area_and_centroid(dof_handler, bearing.boundary_id, min_coordinate_z, max_coordinate_z);
        const double area_z = std::get<0>(result_1);
        const double k_z = bearing.k_equiv_z / area_z;

        auto result_2 = calculate_boundary_area_and_centroid(dof_handler, bearing.boundary_id, min_coordinate_y, max_coordinate_y);
        const double area_y = std::get<0>(result_2);
        const double k_y = bearing.k_equiv_y / area_y;

        boundaryIdToValuesMap[bearing.boundary_id] = std::make_pair(k_z, k_y);
    }

    return boundaryIdToValuesMap;
}

struct Shaft{
    double start_position;
    double end_position;
    double radius_shaft;
    unsigned int boundary_id;
    double density;
    double mass_moment_of_inertia;
    double mass;
    double length;
    double volume;
    double area;
    Point<3> cog;

};
std::vector<Shaft> combineShafts(const Shaft& shaft1, const Shaft& shaft2, const Shaft& shaft3, const Shaft& shaft4) {
    std::vector<Shaft> combinedShafts;

    combinedShafts.push_back(shaft1);
    combinedShafts.push_back(shaft2);
    combinedShafts.push_back(shaft3);
    combinedShafts.push_back(shaft4);

    return combinedShafts;
}
std::map<int, Shaft> combineShaftsToMap(const Shaft& shaft1, const Shaft& shaft2, const Shaft& shaft3, const Shaft& shaft4) {
    std::map<int, Shaft> shaftMap;

    shaftMap[shaft1.boundary_id] = shaft1;
    shaftMap[shaft2.boundary_id] = shaft2;
    shaftMap[shaft3.boundary_id] = shaft3;
    shaftMap[shaft4.boundary_id] = shaft4;

    return shaftMap;
}
//template <int dim>
//int getBoundaryIdForXCoordinateBearing(const std::vector<Bearing>& bearings, double xCoordinate) {
//    for (const auto& bearing : bearings) {
//        if (xCoordinate >= bearing.start_position && xCoordinate <= bearing.end_position) {
//            return bearing.boundary_id;
//        }
//    }
//    return -1; // Return an invalid boundary ID if no matching bearing is found
//}
template <int dim>
int getBoundaryIdForXCoordinateBearing(const std::map< int, Bearing>& bearingsMap, double xCoordinate) {
    for (const auto& bearingPair : bearingsMap) {
        const Bearing& bearing = bearingPair.second;
        if (xCoordinate >= bearing.start_position && xCoordinate <= bearing.end_position) {
            return bearing.boundary_id;
        }
    }
    return -1; // Return an invalid boundary ID if no matching bearing is found
}

//template <int dim>
//int getBoundaryIdForXCoordinateShaft(const std::vector<Shaft>& shafts, double xCoordinate) {
//    for (const auto& shaft : shafts) {
//        if (xCoordinate >= shaft.start_position && xCoordinate <= shaft.end_position) {
//            return shaft.boundary_id;
//        }
//    }
//    return -1; // Return an invalid boundary ID if no matching shaft is found
//}
template <int dim>
int getBoundaryIdForXCoordinateShaft(const std::map< int, Shaft>& shafts, double xCoordinate) {
    for (const auto& shaftPair : shafts) {
        const Shaft& shaft = shaftPair.second;
        if (xCoordinate >= shaft.start_position && xCoordinate <= shaft.end_position) {
            return shaft.boundary_id;
        }
    }
    return -1; // Return an invalid boundary ID if no matching shaft is found
}
template <int dim>
Shaft createShaft(bool isUsed, 
                    double rearmostPoint, 
                    double length, 
                    double radius, 
                    double density, 
                    double mass_moment_of_inertia,
                    unsigned int boundaryId,
                    const DoFHandler<dim> &dof_handler,
                    const Quadrature<dim> &quadrature_formula ) {
    Shaft shaft;

    if (!isUsed) {
        // If the shaft is not used, return an empty or default-initialized Shaft object
        return shaft;
    }

    // Populate the Shaft struct
    shaft.start_position = rearmostPoint;
    shaft.end_position = rearmostPoint + length;
    if (radius==0)
        shaft.radius_shaft = calculate_cylinder_radius(length, mass_moment_of_inertia, density);
    else
        shaft.radius_shaft=radius;
    

    shaft.boundary_id = boundaryId;
    shaft.length = length;
    if (density>0)
        shaft.density = density;
    else    
        shaft.density = calculate_cylinder_density(length, mass_moment_of_inertia, radius);
    if (mass_moment_of_inertia==0)
        shaft.mass_moment_of_inertia = calculate_mass_moment_of_inertia<dim>(dof_handler, Functions::ConstantFunction<dim>(shaft.density), rearmostPoint, rearmostPoint + length, quadrature_formula);
    else
        shaft.mass_moment_of_inertia = mass_moment_of_inertia;
  

    // Calculate other properties
    shaft.volume = M_PI * std::pow(radius, 2) * length;
    shaft.area = 2 * M_PI * radius * length;
    shaft.mass = density * shaft.volume;
    //shaft.mass_moment_of_inertia = shaft.mass * std::pow(radius, 2) / 2;
    //shaft.polar_moment_of_inertia = shaft.mass * std::pow(radius, 2);

    // Calculate centroids
    shaft.cog = Point<3>(rearmostPoint + length / 2, 0, 0);
    
    return shaft;
}
template <int dim>
std::map<int, std::pair<double, Point<dim>>> getAllBearingSpringAreaZ(const std::map<int, Bearing>& bearings, const DoFHandler<dim> &dof_handler) {
    std::map<int, std::pair<double, Point<dim>>> boundaryIdToValuesMap;

    for (const auto& bearingEntry : bearings) {
        const auto& bearing = bearingEntry.second; // Access the Bearing object

        const Point<dim> max_coordinate_z(bearing.start_position, bearing.radius_shaft, -bearing.radius_shaft * 0.8);
        const Point<dim> min_coordinate_z(bearing.end_position, -bearing.radius_shaft, -bearing.radius_shaft);
        
        auto result = calculate_boundary_area_and_centroid(
                dof_handler,
                bearing.boundary_id,
                min_coordinate_z,
                max_coordinate_z);
        
        boundaryIdToValuesMap[bearing.boundary_id] = result;
    }

    return boundaryIdToValuesMap;
}



template <int dim>
Point<dim> get_point_displacement(const DoFHandler<dim> &dof_handler,
                                  const PETScWrappers::MPI::Vector &displacement_vector,
                                  const typename DoFHandler<dim>::active_cell_iterator &cell,
                                  const unsigned int vertex_index) {
    Point<dim> displacement;



    // Extract the displacement for each dimension
    for (unsigned int i = 0; i < dim; ++i) {
        // Get the global DoF index for the i-th displacement component of the vertex
        const types::global_dof_index dof_index = cell->vertex_dof_index(vertex_index, i);
        displacement[i] = displacement_vector(dof_index);
    }


    return displacement;
}
  template<int dim>
  class PropellerForce : public Function<dim>
  {
    public:
      PropellerForce(const double pressure);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override;
      virtual void 
      vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;
      
    private:
      const double pressure;  
  };

  template <int dim>
  PropellerForce<dim>::PropellerForce(const double pressure)
    : Function<dim>(dim), pressure(pressure)
  {}

  template <int dim>
  inline void PropellerForce<dim>::vector_value(const Point<dim> &/*p*/,
                                                Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);
    values = 0;
    values(0) = -pressure;

  }

  template <int dim>
  inline void PropellerForce<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list) const
  {
    const unsigned int n_points = points.size();
 
    AssertDimension(value_list.size(), n_points);
 
    for (unsigned int p = 0; p < n_points; ++p)
      PropellerForce<dim>::vector_value(points[p], value_list[p]);
  }

  template<int dim>
  class PropellerTorque : public Function<dim>
  {
    public:
      PropellerTorque(const double torque, const double polar_moment, const double radius);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override;
      virtual void 
      vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;
      
    private:
      const double torque; 
      const double polar_moment;
      const double radius;
  };

  template <int dim>
  PropellerTorque<dim>::PropellerTorque(const double torque, const double polar_moment, const double radius)
    : Function<dim>(dim), torque(torque), polar_moment(polar_moment), radius(radius)
  {}

  template <int dim>
  inline void PropellerTorque<dim>::vector_value(const Point<dim> &p,
                                                Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);
    double y = p(1);
    double z = p(2);
    double distance = sqrt(pow(y,2)+pow(z,2));
    if (distance<=0)
        distance=0;
    double angle = atan2(z,y);
    //if (angle
    double tau = -3/2*torque/(pow(radius,3)*pi);
    if (isnan(tau))
        tau=0;
     if (isnan(angle))
        {
            angle=0;
            tau=0;
        }
    
    values = 0;
    values(1) = tau* sin(angle);
    values(2) = -tau* cos(angle);

  }

  template <int dim>
  inline void PropellerTorque<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list) const
  {
    const unsigned int n_points = points.size();
 
    AssertDimension(value_list.size(), n_points);
 
    for (unsigned int p = 0; p < n_points; ++p)
      PropellerTorque<dim>::vector_value(points[p], value_list[p]);
  }




  template<int dim>
  class AccelerationAngular : public Function<dim>
  {
    public:
      AccelerationAngular(const double omega);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override;
      virtual void 
      vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;
      
    private:
      const double omega; 

     
  };

  template <int dim>
  AccelerationAngular<dim>::AccelerationAngular(const double omega)
    : Function<dim>(dim), omega(omega)
  {}

  template <int dim>
  inline void AccelerationAngular<dim>::vector_value(const Point<dim> &p,
                                                Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);
    double y = p(1);
    double z = p(2);
    double distance = sqrt(y*y+z*z);
    double angle = atan2(z,y);
    values = 0;
    values(1) = -omega*omega*distance*cos(angle);
    values(2) = -omega*omega*distance*sin(angle);

  }

  template <int dim>
  inline void AccelerationAngular<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list) const
  {
    const unsigned int n_points = points.size();
 
    AssertDimension(value_list.size(), n_points);
 
    for (unsigned int p = 0; p < n_points; ++p)
      AccelerationAngular<dim>::vector_value(points[p], value_list[p]);
  }
  template<int dim>
  class VelocityAngular : public Function<dim>
  {
    public:
      VelocityAngular(const double omega);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override;
      virtual void 
      vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;
      
    private:
      const double omega; 

      
     
  };

  template <int dim>
  VelocityAngular<dim>::VelocityAngular(const double omega)
    : Function<dim>(dim), omega(omega)
  {}

  template <int dim>
  inline void VelocityAngular<dim>::vector_value(const Point<dim> &p,
                                                Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);
    double y = p(1);
    double z = p(2);
    double distance = sqrt(y*y+z*z);
    double angle = atan2(z,y);
    values = 0;
    values(1) = omega*distance*sin(angle);
    values(2) = -omega*distance*cos(angle);

  }

  template <int dim>
  inline void VelocityAngular<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  value_list) const
  {
    const unsigned int n_points = points.size();
 
    AssertDimension(value_list.size(), n_points);
 
    for (unsigned int p = 0; p < n_points; ++p)
      VelocityAngular<dim>::vector_value(points[p], value_list[p]);
  }

  template<int dim>
class AccellerationAngularBoundary : public Function<dim>
{
public:
    AccellerationAngularBoundary(const double omega);

    virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
    virtual void 
    vector_value_list(const std::vector<Point<dim>> &points,
                    std::vector<Vector<double>> &  value_list) const override;
    
private:
    const double omega; 
    
};

template <int dim>
AccellerationAngularBoundary<dim>::AccellerationAngularBoundary(const double omega)
: Function<dim>(dim), omega(omega)
{}

template <int dim>
void AccellerationAngularBoundary<dim>::vector_value(const Point<dim> &p,
                                            Vector<double> &values) const
{
AssertDimension(values.size(), dim);
double y = p(1);
double z = p(2);
double distance = sqrt(y*y+z*z);
double angle = atan2(z,y);

values = 0;
values(1) = -omega * omega *distance*cos(angle);
values(2) = -omega * omega *distance*sin(angle);

}

template <int dim>
 void AccellerationAngularBoundary<dim>::vector_value_list(
const std::vector<Point<dim>> &points,
std::vector<Vector<double>> &  value_list) const
{
const unsigned int n_points = points.size();

AssertDimension(value_list.size(), n_points);

for (unsigned int p = 0; p < n_points; ++p)
    AccellerationAngularBoundary<dim>::vector_value(points[p], value_list[p]);
}

  template<int dim>
class DisplacementEngine : public Function<dim>
{
public:
    DisplacementEngine(const double omega,const double dt);

    virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
    virtual void 
    vector_value_list(const std::vector<Point<dim>> &points,
                    std::vector<Vector<double>> &  value_list) const override;
    
private:
    const double omega; 
    const double dt;
    
};

template <int dim>
DisplacementEngine<dim>::DisplacementEngine(const double omega, const double dt)
: Function<dim>(dim), omega(omega), dt(dt)
{}

template <int dim>
void DisplacementEngine<dim>::vector_value(const Point<dim> &p,
                                            Vector<double> &values) const
{
AssertDimension(values.size(), dim);
double y = p(1);
double z = p(2);
//double distance = sqrt(y*y+z*z);
double angle = atan2(z,y);
double dphi = omega*dt;//+angle;

values = 0;
values(1) = y*cos(dphi) - z*sin(dphi)-y;
values(2) = y*sin(dphi) + z*cos(dphi)-z;

}

template <int dim>
 void DisplacementEngine<dim>::vector_value_list(
const std::vector<Point<dim>> &points,
std::vector<Vector<double>> &  value_list) const
{
const unsigned int n_points = points.size();

AssertDimension(value_list.size(), n_points);

for (unsigned int p = 0; p < n_points; ++p)
    DisplacementEngine<dim>::vector_value(points[p], value_list[p]);
}


template<int dim>
class BearingSurfaceMove : public Function<dim>
{
public:
    BearingSurfaceMove(const double bearing_radius, const double eccentricity , const double phi, const double displacement_increm, const int direction, const double delta_t);

    virtual void h_vector_value(const Point<dim> &p,
                            Vector<double> &values_h) const ;
    virtual void 
    h_vector_value_list(const std::vector<Point<dim>> &points,
                    std::vector<Vector<double>> &  value_h_list) const ;
    
    virtual void h_dot_vector_value(const Point<dim> &p,
                            Vector<double> &values_h_dot) const ;
    virtual void 
    h_dot_vector_value_list(const std::vector<Point<dim>> &points,
                    std::vector<Vector<double>> &  value_h_dot_list) const ;
    
private:
    const double bearing_radius; 
    const double eccentricity;
    const double phi;
    const double displacement_increm;
    const int direction;
    const double delta_t;
   
    
};

template <int dim>
BearingSurfaceMove<dim>::BearingSurfaceMove(const double bearing_radius, const double eccentricity, const double phi, const double displacement_increm, const int direction, const double delta_t)
: Function<dim>(dim), bearing_radius(bearing_radius), eccentricity(eccentricity), phi(phi), displacement_increm(displacement_increm), direction(direction), delta_t(delta_t)
{}

template <int dim>
void BearingSurfaceMove<dim>::h_vector_value(const Point<dim> &p,
                                            Vector<double> &values_h) const
{
AssertDimension(values_h.size(), dim);
    double y = p(1);
    double z = p(2);
    double xb = p(0);
    double yb = eccentricity*cos(phi);
    double zb = eccentricity*sin(phi);
    double y_dot = 0;
    double z_dot = 0;
    if (direction ==1)
    {
        yb = yb + displacement_increm;
        y_dot = displacement_increm/delta_t;

    }
    else if (direction ==2)
    {
        zb = zb + displacement_increm;
        z_dot = displacement_increm/delta_t;
    }

    double x = xb;
    double ins_sqrt = pow((y-yb),2)+pow((z-zb),2);
    double dist = -sqrt(ins_sqrt)+bearing_radius;
  values_h(0) = dist;
  values_h(1) = dist; // -(y-(yb+sqrt(pow(bearing_radius,2)-pow(z-zb,2)))); //h
  values_h(2) = dist;    //  -(z-(zb+sqrt(pow(bearing_radius,2)-pow(y-yb,2))));

  if (dist<0.0)
  {
    values_h(0) = 0.0;

    values_h(1) = 0.0;
    values_h(2) = 0.0;
  }
  



}
template <int dim>
void BearingSurfaceMove<dim>::h_dot_vector_value(const Point<dim> &p,
                                            Vector<double> &values_h_dot) const
{
    AssertDimension(values_h_dot.size(), dim);
    double y = p(1);
    double z = p(2);
    double xb = p(0);
    double yb = eccentricity*cos(phi);
    double zb = eccentricity*sin(phi);
    double y_dot = 0;
    double z_dot = 0;
    if (direction ==1)
    {
        yb = yb + displacement_increm;
        y_dot = displacement_increm/delta_t;

    }
    else if (direction ==2)
    {
        zb = zb + displacement_increm;
        z_dot = displacement_increm/delta_t;
    }

    double x = xb;
    double ins_sqrt = pow((y-yb),2)+pow((z-zb),2);
    double ins_sqrt_dot = -(y-yb)*y_dot-(z-zb)*z_dot;
    double dist = -sqrt(ins_sqrt)+bearing_radius;
    double dist_dot = ins_sqrt_dot/(2*sqrt(ins_sqrt));

    values_h_dot(0) = dist_dot;//0;
    values_h_dot(1) = dist_dot;//-y_dot - (y_dot*(y-yb))/sqrt(pow(bearing_radius,2)-pow(z-zb,2));
    values_h_dot(2) = dist_dot;//-z_dot - (z_dot*(z-zb))/sqrt(pow(bearing_radius,2)-pow(y-yb,2));

}

template <int dim>
 void BearingSurfaceMove<dim>::h_vector_value_list(
const std::vector<Point<dim>> &points,
std::vector<Vector<double>> &  value_h_list) const
{
const unsigned int n_points = points.size();

AssertDimension(value_h_list.size(), n_points);

for (unsigned int p = 0; p < n_points; ++p)
    BearingSurfaceMove<dim>::h_vector_value(points[p], value_h_list[p]);
}

template <int dim>
 void BearingSurfaceMove<dim>::h_dot_vector_value_list(
const std::vector<Point<dim>> &points,
std::vector<Vector<double>> &  value_h_dot_list) const
{
const unsigned int n_points = points.size();

AssertDimension(value_h_dot_list.size(), n_points);

for (unsigned int p = 0; p < n_points; ++p)
    BearingSurfaceMove<dim>::h_dot_vector_value(points[p], value_h_dot_list[p]);
}



//
  template <int dim>
  TopLevel<dim>::TopLevel(const std::string &input_file)
    : prm(input_file)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , triangulation(MPI_COMM_WORLD /*, 
                    typename Triangulation<dim>::MeshSmoothing(
                        Triangulation<dim>::smoothing_on_refinement |
                        Triangulation<dim>::smoothing_on_coarsening 
                      )
                     ,typename Triangulation<dim>::Triangulationcheck_for_distorted_cells =true*/)

                        
    , fe(FE_Q<dim>(1), dim)
    , dof_handler(triangulation)
    , quadrature_formula(fe.degree + 1)
    //, face_quadrature_formula(fe_values.get_fe().degree + 1)
    , face_quadrature_formula(fe.degree + 1)
    , present_time(0)
    , present_timestep(prm.delta_t)
    , end_time(prm.end_time)
    , timestep_no(0)

    , pcout(std::cout,
              (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                        pcout,
                        TimerOutput::never,
                        TimerOutput::wall_times)
  {}
 
 
 
  template <int dim>
  TopLevel<dim>::~TopLevel()
  {
    dof_handler.clear();
  }
 
 
 
  template <int dim>
  void TopLevel<dim>::run()
  {
    do_initial_timestep();
    computing_timer.print_summary();
    computing_timer.reset();

    while (present_time < end_time)
       {
      do_timestep();        
      computing_timer.print_summary();
      computing_timer.reset();
      pcout << std::endl;
       }
  }

 
 //TODO CHECK THE BOUNDARIES

  template <int dim>
  void TopLevel<dim>::create_coarse_grid()
  {
    TimerOutput::Scope t(computing_timer, "create_coarse_grid");

    parallel::shared::Triangulation<dim> triangulation_intermediate_shaft(MPI_COMM_WORLD);
    parallel::shared::Triangulation<dim> triangulation_thrust_shaft(MPI_COMM_WORLD);
    parallel::shared::Triangulation<dim> triangulation_engine_shaft(MPI_COMM_WORLD);
    parallel::shared::Triangulation<dim> triangulation_propeller_shaft(MPI_COMM_WORLD);
    unsigned int partitions;
    unsigned int partitions_divider=1;
    std::vector<parallel::shared::Triangulation<dim>*>triangulation_vector;
    if (prm.useShaft_thrustshaft)
        partitions_divider+=1;
    if (prm.useShaft_engineshaft)
        partitions_divider+=1;
    if (prm.useShaft_propellershaft)
        partitions_divider+=1;
//
    double minus_half_length;

    double plus_half_length;
    if(prm.useShaft_propellershaft) 
         minus_half_length = prm.distance_from_point_propellershaft;
    else 
         minus_half_length = prm.distance_from_point_intermediateshaft;

    if (prm.useShaft_engineshaft)
         plus_half_length = prm.distance_from_point_engineshaft + prm.length_engineshaft;
    else if (prm.useShaft_thrustshaft)
         plus_half_length = prm.distance_from_point_thrustshaft + prm.length_thrustshaft;
    else
         plus_half_length = prm.distance_from_point_intermediateshaft + prm.length_intermediateshaft;
    partitions = ceilDiv(prm.partitions,partitions_divider);
    if (prm.useShaft_intermediateshaft)
    {
        if (prm.radius_intermediateshaft==0 && prm.mass_inertia_intermediateshaft>0)
        {
            double radius_intermediateshaft = calculate_cylinder_radius(prm.length_intermediateshaft, prm.mass_inertia_intermediateshaft, prm.rho_intermediateshaft);
            prm.radius_intermediateshaft = radius_intermediateshaft;
        }
        GridGenerator::subdivided_cylinder(triangulation_intermediate_shaft, partitions, prm.radius_intermediateshaft, prm.length_intermediateshaft/2);
        GridTools::partition_triangulation (n_mpi_processes,
                                           triangulation_intermediate_shaft);
       const double x0 = -prm.length_intermediateshaft/2;
       const double x1 =  prm.length_intermediateshaft/2;
       const double dL = (x1 - x0) / n_mpi_processes;
       {
       typename Triangulation<dim>::active_cell_iterator
           cell = triangulation_intermediate_shaft.begin_active(),
           endc = triangulation_intermediate_shaft.end();
       for (; cell != endc; ++cell)
           {
           const dealii::Point<dim> &center = cell->center();
           const double              x      = center[0];
           const unsigned int id = std::floor((x - x0) / dL);
           cell->set_subdomain_id(id);
           }
       }
        Tensor< 1, dim >  	shift_vector;
        shift_vector[0] = prm.length_intermediateshaft/2+prm.distance_from_point_intermediateshaft;
        GridTools::shift(shift_vector,triangulation_intermediate_shaft)	;
        triangulation_vector.push_back(&triangulation_intermediate_shaft);
        //GridGenerator::merge_triangulations(triangulation_intermediate_shaft,
        //                            triangulation,
        //                            triangulation,
        //                            1.0e-16,
        //                            false,
        //                            true);
    }
    if (prm.useShaft_thrustshaft)
    {
        if (prm.radius_thrustshaft==0&&prm.mass_inertia_thrustshaft>0)
        {
            double radius_thrustshaft = calculate_cylinder_radius(prm.length_thrustshaft, prm.mass_inertia_thrustshaft, prm.rho_thrustshaft);
            prm.radius_thrustshaft = radius_thrustshaft;
        }
        GridGenerator::subdivided_cylinder(triangulation_thrust_shaft, partitions, prm.radius_thrustshaft, prm.length_thrustshaft/2);
        GridTools::partition_triangulation (n_mpi_processes,
                                           triangulation_thrust_shaft);
        const double x0 = -prm.length_thrustshaft/2;
        const double x1 =  prm.length_thrustshaft/2;
        const double dL = (x1 - x0) / n_mpi_processes;
        {
        typename Triangulation<dim>::active_cell_iterator
            cell = triangulation_thrust_shaft.begin_active(),
            endc = triangulation_thrust_shaft.end();
        for (; cell != endc; ++cell)
            {
            const dealii::Point<dim> &center = cell->center();
            const double              x      = center[0];
            const unsigned int id = std::floor((x - x0) / dL);
            cell->set_subdomain_id(id);
            }
        }
        Tensor< 1, dim >  	shift_vector;
        shift_vector[0] = prm.length_thrustshaft/2+prm.distance_from_point_thrustshaft;
        GridTools::shift(shift_vector,triangulation_thrust_shaft)	;
        triangulation_vector.push_back(&triangulation_thrust_shaft);

        
       // GridGenerator::merge_triangulations(triangulation_thrust_shaft,
       //                             triangulation,
       //                             triangulation,
       //                             1.0e-16,
       //                             false,
       //                             true);

    }
    if (prm.useShaft_engineshaft)
    {
        if (prm.radius_engineshaft==0&&prm.mass_inertia_engineshaft>0)
        {
            double radius_engineshaft = calculate_cylinder_radius(prm.length_engineshaft, prm.mass_inertia_engineshaft, prm.rho_engineshaft);
            prm.radius_engineshaft = radius_engineshaft;
        }
        GridGenerator::subdivided_cylinder(triangulation_engine_shaft, partitions, prm.radius_engineshaft, prm.length_engineshaft/2);
        GridTools::partition_triangulation (n_mpi_processes,
                                           triangulation_engine_shaft);
        const double x0 = -prm.length_engineshaft/2;
        const double x1 =  prm.length_engineshaft/2;
        const double dL = (x1 - x0) / n_mpi_processes;
        {
        typename Triangulation<dim>::active_cell_iterator
            cell = triangulation_engine_shaft.begin_active(),
            endc = triangulation_engine_shaft.end();
        for (; cell != endc; ++cell)
            {
            const dealii::Point<dim> &center = cell->center();
            const double              x      = center[0];
            const unsigned int id = std::floor((x - x0) / dL);
            cell->set_subdomain_id(id);
            }
        }
        Tensor< 1, dim >  	shift_vector;
        shift_vector[0] = prm.length_engineshaft/2+prm.distance_from_point_engineshaft;
        GridTools::shift(shift_vector,triangulation_engine_shaft)	;
        triangulation_vector.push_back(&triangulation_engine_shaft);
        //GridGenerator::merge_triangulations(triangulation_engine_shaft,
        //                            triangulation,
        //                            triangulation,
        //                            1.0e-16,
        //                            false,
        //                            true);

    }
    
    if (prm.useShaft_propellershaft)
    {
        if (prm.radius_propellershaft==0&&prm.mass_inertia_propellershaft>0)
        {
            double radius_propellershaft = calculate_cylinder_radius(prm.length_propellershaft, prm.mass_inertia_propellershaft, prm.rho_propellershaft);
            prm.radius_propellershaft = radius_propellershaft;
        }
        GridGenerator::subdivided_cylinder(triangulation_propeller_shaft, partitions, prm.radius_propellershaft, prm.length_propellershaft/2);
        GridTools::partition_triangulation (n_mpi_processes,
                                           triangulation_propeller_shaft);
        const double x0 = -prm.length_propellershaft/2;
        const double x1 =  prm.length_propellershaft/2;
        const double dL = (x1 - x0) / n_mpi_processes;
        {
        typename Triangulation<dim>::active_cell_iterator
            cell = triangulation_propeller_shaft.begin_active(),
            endc = triangulation_propeller_shaft.end();
        for (; cell != endc; ++cell)
            {
            const dealii::Point<dim> &center = cell->center();
            const double              x      = center[0];
            const unsigned int id = std::floor((x - x0) / dL);
            cell->set_subdomain_id(id);
            }
        }
        Tensor< 1, dim >  	shift_vector;
        shift_vector[0] = prm.length_propellershaft/2+prm.distance_from_point_propellershaft;
        GridTools::shift(shift_vector,triangulation_propeller_shaft)	;
        triangulation_vector.push_back(&triangulation_propeller_shaft);
        //GridGenerator::merge_triangulations(triangulation_propeller_shaft,
        //                            triangulation,
        //                            triangulation,
        //                            1.0e-16,
        //                            false,
        //                            true);
    }
    if(!prm.useShaft_engineshaft && !prm.useShaft_intermediateshaft &&!prm.useShaft_thrustshaft &&!prm.useShaft_propellershaft)
    {

        GridGenerator::subdivided_cylinder(triangulation, partitions, prm.radius_intermediateshaft, prm.length_intermediateshaft/2);
        GridTools::partition_triangulation (n_mpi_processes,
                                           triangulation);
        const double x0 = -prm.length_intermediateshaft/2;
        const double x1 =  prm.length_intermediateshaft/2;
        const double dL = (x1 - x0) / n_mpi_processes;
        {
        typename Triangulation<dim>::active_cell_iterator
            cell = triangulation.begin_active(),
            endc = triangulation.end();
        for (; cell != endc; ++cell)
            {
            const dealii::Point<dim> &center = cell->center();
            const double              x      = center[0];
            const unsigned int id = std::floor((x - x0) / dL);
            cell->set_subdomain_id(id);
            }
        }
        Tensor< 1, dim >  	shift_vector;
        shift_vector[0] = prm.length_intermediateshaft/2+prm.distance_from_point_intermediateshaft;
        GridTools::shift(shift_vector,triangulation);
    }

    else
    {
        
        if (prm.useShaft_engineshaft)
            GridGenerator::merge_triangulations(triangulation_intermediate_shaft,
                                    triangulation_engine_shaft,
                                    triangulation,
                                    1.0e-12,
                                    false,
                                    true);
        //GridTools::partition_triangulation (n_mpi_processes,
        //                                   triangulation);
        //const double x0 = minus_half_length;
        //const double x1 =  plus_half_length;
        //const double dL = (x1 - x0) / n_mpi_processes;
        //{
        //typename Triangulation<dim>::active_cell_iterator
        //    cell = triangulation.begin_active(),
        //    endc = triangulation.end();
        //for (; cell != endc; ++cell)
        //    {
        //    const dealii::Point<dim> &center = cell->center();
        //    const double              x      = center[0];
        //    const unsigned int id = std::floor((x - x0) / dL);
        //    cell->set_subdomain_id(id);
        //    }
        //}
        const Tensor<1, dim>           axis({1.0, 0.0, 0.0});
        const Point<dim>               axial_point(0.0, 0.0, 0.0);
        const CylindricalManifold<dim> cylinder(axis, axial_point);
        const types::manifold_id     cylinder_id = 10;
        triangulation.set_manifold(cylinder_id, cylinder);
        for (auto &cell : triangulation.active_cell_iterators())
            //if (cell->center()[1] >= 3.0)
                cell->set_all_manifold_ids(cylinder_id);
        for (const auto &cell : triangulation.active_cell_iterators())
        for (const auto &face : cell->face_iterators())
            {
            const Point<3> face_center = face->center();
            if (std::abs(face_center[1]) < 1.0e-5 &&
                std::abs(face_center[2]) < 1.0e-5)
                cell->set_all_manifold_ids(numbers::flat_manifold_id);
            }

    }
    triangulation.refine_global(prm.n_global_refinements); 
    if (prm.change_diameter)
     {
      change_diameter(prm.x_1,prm.x_2,prm.radius_intermediateshaft, prm.radius_final);
      triangulation.execute_coarsening_and_refinement();
     }
    //
    if (prm.remove_hanging_nodes)
        GridTools::remove_hanging_nodes(triangulation,true,100);

    
    journal_bearings   = calculateBearingsMap(prm.bearings_journal,   
                                            prm.distance_between_bearings_journal,      
                                            prm.length_journal,   
                                            prm.distance_journal,   
                                            prm.radius_intermediateshaft,   
                                            prm.radius_journal,   
                                            prm.eccentricity_journal,
                                            prm.phi_journal,
                                            prm.k_equiv_z_journal,
                                            prm.k_equiv_y_journal,
                                            prm.friction_coeff_journal,
                                            prm.viscocity_journal,
                                            prm.displacement_journal_no1_ID,
                                            prm.displacement_journal_no1_displacement,
                                            prm.displacement_journal_no1_direction,
                                            prm.displacement_journal_no1_endtime,
                                            prm.displacement_journal_no2_ID,
                                            prm.displacement_journal_no2_displacement,
                                            prm.displacement_journal_no2_direction,
                                            prm.displacement_journal_no2_endtime,
                                            7);
    tailtube_bearings  = calculateBearingsMap(prm.bearings_tailtube,  
                                            prm.distance_between_bearings_tailtube,     
                                            prm.length_tailtube,   
                                            prm.distance_tailtube,  
                                            prm.radius_propellershaft,      
                                            prm.radius_tailtube, 
                                            prm.eccentricity_tailtube,
                                            prm.phi_tailtube,
                                            prm.k_equiv_z_tailtube,
                                            prm.k_equiv_y_tailtube,
                                            prm.friction_coeff_tailtube,
                                            prm.viscocity_tailtube,
                                            prm.displacement_tailtube_no1_ID,
                                            prm.displacement_tailtube_no1_displacement,
                                            prm.displacement_tailtube_no1_direction,
                                            prm.displacement_tailtube_no1_endtime,
                                            prm.displacement_tailtube_no2_ID,
                                            prm.displacement_tailtube_no2_displacement,
                                            prm.displacement_tailtube_no2_direction,
                                            prm.displacement_tailtube_no2_endtime,
                                            17);
    sterntube_bearings = calculateBearingsMap(prm.bearings_sterntube, 
                                            prm.distance_between_bearings_sterntube,    
                                            prm.length_sterntube,  
                                            prm.distance_sterntube, 
                                            prm.radius_propellershaft,      
                                            prm.radius_sterntube, 
                                            prm.eccentricity_sterntube,
                                            prm.phi_sterntube,
                                            prm.k_equiv_z_sterntube,
                                            prm.k_equiv_y_sterntube,
                                            prm.friction_coeff_sterntube,
                                            prm.viscocity_sterntube,
                                            prm.displacement_sterntube_no1_ID,
                                            prm.displacement_sterntube_no1_displacement,
                                            prm.displacement_sterntube_no1_direction,
                                            prm.displacement_sterntube_no1_endtime,
                                            prm.displacement_sterntube_no2_ID,
                                            prm.displacement_sterntube_no2_displacement,
                                            prm.displacement_sterntube_no2_direction,
                                            prm.displacement_sterntube_no2_endtime,                                            
                                            27);
                                              
    engine_bearings    = calculateBearingsMap(prm.bearings_engine,    
                                            prm.distance_between_bearings_engine,       
                                            prm.length_engine,     
                                            prm.distance_engine,    
                                            prm.radius_thrustshaft,         
                                            prm.radius_engine,   
                                            prm.eccentricity_engine,
                                            prm.phi_engine,
                                            prm.k_equiv_z_engine,
                                            prm.k_equiv_y_engine,
                                            prm.friction_coeff_engine,
                                            prm.viscocity_engine,
                                            prm.displacement_engine_no1_ID,
                                            prm.displacement_engine_no1_displacement,
                                            prm.displacement_engine_no1_direction,
                                            prm.displacement_engine_no1_endtime,
                                            prm.displacement_engine_no2_ID,
                                            prm.displacement_engine_no2_displacement,
                                            prm.displacement_engine_no2_direction,
                                            prm.displacement_engine_no2_endtime,
                                            37);

    std::map<int, Bearing> allBearings = combineBearingsMap(journal_bearings, tailtube_bearings, sterntube_bearings, engine_bearings);


    

    Tensor<1,dim> dist_vector;
    Point<dim> center(0, 0, 0);
    typename Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (; cell != endc; ++cell)
       for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
        if (cell->face(f)->at_boundary())
          {
            
            dist_vector = cell->face(f)->center() - center;
            double x_coord = dist_vector[0];
            if (dist_vector[0] == plus_half_length)
               {
                cell->face(f)->set_boundary_id(0);
               }
            else if (dist_vector[0] == minus_half_length)
               {
                cell->face(f)->set_boundary_id(1);
               }
            else 
            {
                if (prm.useShaft_intermediateshaft)
                {
                    if (dist_vector[0] >= prm.distance_from_point_intermediateshaft && dist_vector[0] <= prm.distance_from_point_intermediateshaft+prm.length_intermediateshaft)
                    {
                        cell->face(f)->set_boundary_id(2);
                    }
                }
                else if (prm.useShaft_thrustshaft)
                {
                    if (dist_vector[0] >= prm.distance_from_point_thrustshaft&& dist_vector[0] <= prm.distance_from_point_thrustshaft+prm.length_thrustshaft)
                    {
                        cell->face(f)->set_boundary_id(3);
                    }
                }
                else if (prm.useShaft_engineshaft)
                {
                    if (dist_vector[0] >= prm.distance_from_point_engineshaft&& dist_vector[0] <= prm.distance_from_point_engineshaft+prm.length_engineshaft)
                    {
                        cell->face(f)->set_boundary_id(4);
                    }
                }
                else if (prm.useShaft_propellershaft)
                {
                    if (dist_vector[0] >= prm.distance_from_point_propellershaft&& dist_vector[0] <= prm.distance_from_point_propellershaft+prm.length_propellershaft)
                    {
                        cell->face(f)->set_boundary_id(5);
                    }
                }
                else if (dist_vector[0]>=minus_half_length && dist_vector[0]<=minus_half_length-prm.prop_length)
                {
                    cell->face(f)->set_boundary_id(6);
                }
                else
                {
                    cell->face(f)->set_boundary_id(2);
                }
    
                if (getBoundaryIdForXCoordinateBearing<dim>(allBearings, x_coord)>0)
                                {
                                    cell->face(f)->set_boundary_id(getBoundaryIdForXCoordinateBearing<dim>(allBearings, x_coord));
                                }

                
            }
    

          }  
        
        }   
    setup_quadrature_point_history();

  }
 
 
   template <int dim>
  void  TopLevel<dim>::change_diameter(/*Triangulation<dim> &triangulation,*/ const double x_1, const double x_2,const double original_radius, const double radius_final)
  {
       double scaling_factor = radius_final / original_radius;

      std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
        typename Triangulation<dim>::active_cell_iterator
          cell = triangulation.begin_active(),
          endc = triangulation.end();
        for (; cell != endc; ++cell)
          for (const auto v : cell->vertex_indices())
            if (vertex_touched[cell->vertex_index(v)] == false)
              {
                vertex_touched[cell->vertex_index(v)] = true;
                Point<dim> point = triangulation.get_vertices()[cell->vertex_index(v)];
                if (point(0) <x_2 && point(0) >x_1 )
                {
                  double r = std::sqrt(point[0] * point[0] + point[1] * point[1]);
                  double new_r = r * scaling_factor;
                  double dr = new_r - r;
                  Point<dim> vertex_displacement;
                 // for (unsigned int d = 0; d < dim; ++d)
                  vertex_displacement[0] = 0;
                  vertex_displacement[1] =  dr * (point[1] / r);
                  vertex_displacement[2] =  dr * (point[2] / r);

                  cell->vertex(v) += vertex_displacement;
                  //if (abs(point(0)-x_1)<0.1)
                  //  cell->set_refine_flag();
                }
                if (abs(point(0)-x_1)<0.1)
                  cell->set_refine_flag();
              }
        triangulation.execute_coarsening_and_refinement();
    }
  template <int dim>
  void TopLevel<dim>::refine_big_cells()
  {
      double min_cell_diameter = GridTools::minimal_cell_diameter	(triangulation);
      double max_cell_diameter = GridTools::maximal_cell_diameter	(triangulation);
      double min_cell_diameter_ratio = min_cell_diameter/max_cell_diameter;
      double min_cell_diameter_ratio_threshold = 0.85;
      while (min_cell_diameter_ratio<min_cell_diameter_ratio_threshold)
      {
        
      typename Triangulation<dim>::active_cell_iterator
          cell = triangulation.begin_active(),
          endc = triangulation.end();
        for (; cell != endc; ++cell)
        //for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        //  {
         // if (cell->face(f)->at_boundary())
            if (cell-> diameter()> min_cell_diameter_ratio_threshold*max_cell_diameter)
            {
              cell->set_refine_flag();
             // break;
            }
            
          //}
      
      triangulation.execute_coarsening_and_refinement();
     // min_cell_diameter = GridTools::minimal_cell_diameter	(triangulation);
      max_cell_diameter = GridTools::maximal_cell_diameter	(triangulation);
      min_cell_diameter_ratio = min_cell_diameter/max_cell_diameter;
      }
  }

  template <int dim>
  void TopLevel<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");
    
           { TimerOutput::Scope t(computing_timer,"Setup: distribute DoFs");}
   
    

    //distributes the dofs to the processors
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::subdomain_wise(dof_handler);
    std::vector<dealii::IndexSet> locally_owned_dofs_per_processor =
        DoFTools::locally_owned_dofs_per_subdomain(dof_handler);

    locally_owned_dofs = dof_handler.locally_owned_dofs(); //locally_owned_dofs_per_processor[this_mpi_process];

    locally_relevant_dofs = 
           DoFTools::extract_locally_relevant_dofs(dof_handler); //, locally_relevant_dofs);
    
    
   { TimerOutput::Scope t(computing_timer, "Setup: constraints");}

    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
   // hanging_node_constraints.reinit(locally_relevant_dofs);
    hanging_node_constraints.close();

    compute_dirichlet_constraints();

    DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    sparsity_pattern,
                                    constraints_dirichlet_and_hanging_nodes,
                                    /*keep constrained dofs*/ false);



    //DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    std::vector<dealii::types::global_dof_index> n_locally_owned_dofs(n_mpi_processes);

    for (unsigned int i = 0; i < n_mpi_processes; ++i)
      n_locally_owned_dofs[i] = locally_owned_dofs_per_processor[i].n_elements();


    SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                                dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);
    
    //PETSc preallocating the system matrix
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         sparsity_pattern,
                         mpi_communicator);
    tmp_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         sparsity_pattern,
                         mpi_communicator);
    //PETSc correcting the preallocated size of the  righthand side and dx matrices

    system_mass_matrix.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          sparsity_pattern,
                          mpi_communicator);
    
    system_dynamic_matrix.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          sparsity_pattern,
                          mpi_communicator);
    system_damping_matrix.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          sparsity_pattern,
                          mpi_communicator);
    bearing_system_matrix.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          sparsity_pattern,
                          mpi_communicator);
    bearing_rhs.reinit(locally_owned_dofs, mpi_communicator);

    
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs_prev.reinit(locally_owned_dofs, mpi_communicator);

    //locally_relevant_solution.reinit(locally_owned_dofs, mpi_communicator);

    displacement_n.reinit(locally_owned_dofs, mpi_communicator);
    velocity_n.reinit(locally_owned_dofs, mpi_communicator);
    acceleration_n.reinit(locally_owned_dofs, mpi_communicator);

    acceleration_n_prev.reinit(locally_owned_dofs, mpi_communicator);
    velocity_n_prev.reinit(locally_owned_dofs, mpi_communicator);
    displacement_n_prev.reinit(locally_owned_dofs, mpi_communicator);

    newton_rhs.reinit(locally_owned_dofs, mpi_communicator);



    total_displacement.reinit(locally_owned_dofs, mpi_communicator);

   



   
    
   
    

    locally_relevant_solution.reinit(locally_owned_dofs,
                                       locally_relevant_dofs,
                                       mpi_communicator);
    displacement_locally_relevant_solution.reinit(locally_owned_dofs,
                                       locally_relevant_dofs,
                                       mpi_communicator);
    bearing_locally_relevant_solution.reinit(locally_owned_dofs,
                                       locally_relevant_dofs,
                                       mpi_communicator);
    total_bearings_locally_relevant_solution.reinit(locally_owned_dofs,
                                       locally_relevant_dofs,
                                       mpi_communicator);
    total_displacement = 0;
  
    newton_rhs = 0;
    displacement_n = 0;
    velocity_n = 0;
    acceleration_n = 0;
    acceleration_n_prev = 0;
    velocity_n_prev = 0;
    displacement_n_prev = 0;
    system_rhs = 0;
    system_rhs_prev = 0;
    system_matrix = 0;
    tmp_matrix = 0;
    system_mass_matrix = 0;
    system_dynamic_matrix = 0;
    system_damping_matrix = 0;
    bearing_system_matrix = 0;
    bearing_rhs = 0;
    total_bearings_locally_relevant_solution = 0;
    displacement_locally_relevant_solution = 0;
    bearing_locally_relevant_solution = 0;
    locally_relevant_solution = 0;
    //locally_relevant_solution.reinit(dof_handler.n_dofs());
    //                      mpi_communicator);
    eigenfunctions.resize(prm.eigenmodenumber);
    
    //eigenfunctions.resize(5);
    for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
      {
        eigenfunctions[i].reinit(locally_owned_dofs,
                                mpi_communicator); // without ghost dofs
        for (unsigned int j = 0; j < locally_owned_dofs.n_elements(); ++j)
          eigenfunctions[i][locally_owned_dofs.nth_index_in_set(j)] =rand();

        eigenfunctions[i].compress(dealii::VectorOperation::insert);
      }

    eigenvalues.resize(eigenfunctions.size());
  }
 
 
 template <int dim>
 void TopLevel<dim>::compute_dirichlet_constraints()
 {
    constraints_dirichlet_and_hanging_nodes.reinit(locally_relevant_dofs);
    constraints_dirichlet_and_hanging_nodes.merge(hanging_node_constraints);

    std::vector<bool> component_mask(dim);
    if (prm.dynamicmode)
    {
        //component_mask[0] = true;
        //component_mask[1] = true;
        //component_mask[2] = true;
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                      0
        //                                      ,
        //                                     Functions::ZeroFunction<dim>(dim),
        //                                      constraints_dirichlet_and_hanging_nodes,  component_mask);
        //component_mask[0] = false;
        //component_mask[1] = false;
        //component_mask[2] = false;
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                        1
        //                                        ,
        //                                        Functions::ZeroFunction<dim>(dim),
        //                                        constraints_dirichlet_and_hanging_nodes,  component_mask);
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                        1
        //                                        ,
        //                                        Functions::ZeroFunction<dim>(dim),
        //                                        constraints_dirichlet_and_hanging_nodes,  component_mask);
       // VectorTools::interpolate_boundary_values(dof_handler,          
    }
    else
    {
        //component_mask[0] = true;
        //component_mask[1] = true;
        //component_mask[2] = false;
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                        1,
        //                                        Functions::ZeroFunction<dim>(dim),
        //                                        constraints_dirichlet_and_hanging_nodes,
        //                                        component_mask); 
        //component_mask[1] = true;
        //component_mask[2] = false;
        //component_mask[0] = true;
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                        0,
        //                                        Functions::ZeroFunction<dim>(dim),
        //                                        constraints_dirichlet_and_hanging_nodes,
        //                                        component_mask);
        //component_mask[0] = false;
        //component_mask[1] = false;
        //component_mask[2] = true;
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                        4,
        //                                        Functions::ZeroFunction<dim>(dim),
        //                                        constraints_dirichlet_and_hanging_nodes,
        //                                        component_mask);
        //component_mask[0] = false;
        //component_mask[1] = false;
        //component_mask[2] = true;
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                        7,
        //                                        Functions::ZeroFunction<dim>(dim),
        //                                        constraints_dirichlet_and_hanging_nodes,
        //                                        component_mask);
        //                                                                                      
        //component_mask[0] = true;
        //component_mask[1] = false;
        //component_mask[2] = false;
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                        0
        //                                        ,
        //                                        Functions::ZeroFunction<dim>(dim),
        //                                        constraints_dirichlet_and_hanging_nodes,  component_mask);
        //component_mask[0] = true;
        //component_mask[1] = false;
        //component_mask[2] = false;
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                        10
        //                                        ,
        //                                        Functions::ZeroFunction<dim>(dim),
        //                                        constraints_dirichlet_and_hanging_nodes,  component_mask);
        component_mask[0] = true;
        component_mask[1] = true;
        component_mask[2] = true;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                0
                                                ,
                                                Functions::ZeroFunction<dim>(dim),
                                                constraints_dirichlet_and_hanging_nodes,  component_mask);
         
     
    
    }
    constraints_dirichlet_and_hanging_nodes.close();
 }
 
  template <int dim>
  void TopLevel<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");
    system_rhs    = 0;
    system_matrix = 0;
    system_mass_matrix = 0;
    system_dynamic_matrix = 0;
    system_damping_matrix = 0;
   
    
    pcout << "Timestep = "<< timestep_no <<std::endl;
    //if (timestep_no==1)
    //{
    //  displacement_n = 0;
    //  velocity_n = 0;
    //  acceleration_n = 0;
    //}

      
    const double rpm = prm.rpm;
    
   
   
    //BearingForce<dim> bearing_force(rpm, prm.bearing_diameter, prm.viscocity);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values | update_normal_vectors );
    
  

    FEFaceValues<dim> fe_face_values(fe, 
                                            face_quadrature_formula,
                                           update_values | update_quadrature_points | update_JxW_values | update_normal_vectors );
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
    //std::cout << n_q_points << std::endl;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_damping_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    Vector<double>     cell_res(dofs_per_cell);

    Vector<double>     cell_displacement(dofs_per_cell);
    Vector<double>     cell_velocity(dofs_per_cell);
    Vector<double>     cell_accel(dofs_per_cell);
    Vector<double>     cell_accel_rotation(dofs_per_cell);



    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
    

    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell ;
    

    
    
    bool calculate_polar_moment = true; // set to true if you want to calculate the polar moment of inertia
    auto [total_area, centroid_, polar_moment] = calculate_boundary_properties(dof_handler, 1, calculate_polar_moment);     

   // double bearing_area = calculate_boundary_area(dof_handler,3);
    auto [total_area_prop, boundary_centroid] = calculate_boundary_area_and_centroid(dof_handler, 6);
   
   // pcout <<  "\n Total Area of Tailbearing: "<< total_area_prop << std::flush;
    //pcout <<  "\n Centroid of Tailbearing: "<< boundary_centroid << std::flush;
    
    //pcout <<  "\n Total Area of propellershaft: "<< total_area_prop << std::flush;

    const double prop_weight_pressure = prm.prop_weight*prm.g/total_area_prop;
    const double total_force = rpmToForce(rpm, prm.MCR/pow(prm.rpm_MCR*rpm_to_rps,3), prm.v_ship);
    const double total_torque = rpmToTorque(rpm,prm.MCR/pow(prm.rpm_MCR*rpm_to_rps,3));
    pcout <<  "\n Total Torque of propellershaft: "<< total_torque << std::flush;
    const double pressure = total_force / total_area;
  

    //if (polar_moment > 0)
    //  {
    //    polar_moment = polar_moment;
    //  }
    //  else
    //  {
    //    polar_moment = pi*pow(prm.radius,4)/32;
    //  }
    //
    //if (this_mpi_process==0)
    pcout <<  "\n Total Area of section: "<< total_area << "\n Pressure=  " << pressure <<" for RPM="<<prm.rpm << "\n Centroid=  " << centroid_ << "\n Polar Moment of Inertia J=  " << polar_moment<<  std::flush;
    
    //pcout <<  "\n Second Calc from Func: Total Area of section: "<< boundary_area << "\n Pressure=  " << pressure <<" for RPM="<<prm.rpm << "\n Centroid=  " << boundary_centroid_ << "\n Polar Moment of Inertia J=  " << polar_moment_<<  std::flush;

    auto stress_strain_tensor_ = get_stress_strain_tensor<dim>(prm.lambda, prm.mu);

      pcout << "    Starting to  construct matrices for " << this_mpi_process <<" process done." <<std::endl;
    //for (const auto &cell : dof_handler.active_cell_iterators()) 
      //if (cell->is_locally_owned())
    std::map<int,Bearing> allBearings = combineBearingsMap(journal_bearings, tailtube_bearings, sterntube_bearings, engine_bearings);
    


    unsigned int count=0;
    unsigned int count_fe_face_values_nan=0;
    unsigned int count_fe_face_JxW_nan=0;
    unsigned int count_rhs_nan=0;
    unsigned int count_n_faces_2=0;
      BodyForce<dim>              body_force(prm.rho, prm.g);
    std::vector<Vector<double>> body_force_values(n_q_points,
                                                  Vector<double>(dim));
    //double max_z_spring = -0.8*prm.radius;
    //double min_z_spring = -prm.radius;
    //Point<dim> max_spring_point(prm.half_length,prm.radius,max_z_spring);
    //Point<dim> min_spring_point(-prm.half_length,-prm.radius,min_z_spring);
    //auto [journal_bearing_spring_area, journal_bearing_spring_centroid]=calculate_boundary_area_and_centroid_with_constraint(dof_handler, 7, dim-1, min_z_spring, max_z_spring) ;
    
    auto boundaryStiffenessMap= getAllBearingSpringK(allBearings,dof_handler) ;
    auto boundaryAreaZMap= getAllBearingSpringAreaZ(allBearings,dof_handler) ;
    pcout <<  "\n        Area of journal bearing spring: "<< std::get<0>(boundaryAreaZMap[7]) << " with centroid=  " << std::get<1>(boundaryAreaZMap[7]) <<std::flush;
    //std::map<types::boundary_id, double> boundary_area_map;
    //std::map<types::boundary_id, std::pair<double, Point<dim>>> boundary_info;
    //if (timestep_no==1)
    //  calculate_and_store_boundary_areas_and_centroids(
    //    dof_handler, min_spring_point, max_spring_point,true, boundary_info);
    //else
    //  calculate_and_store_boundary_areas_and_centroids(
    //    dof_handler, min_spring_point, max_spring_point,false, boundary_info);   
    
    Shaft intermediate_shaft = createShaft(prm.useShaft_intermediateshaft,
                                            prm.distance_from_point_intermediateshaft,
                                            prm.length_intermediateshaft,
                                            prm.radius_intermediateshaft,
                                            prm.rho_intermediateshaft,
                                            prm.mass_inertia_intermediateshaft,
                                            2,
                                            dof_handler,
                                            quadrature_formula);
    Shaft thrust_shaft = createShaft(prm.useShaft_thrustshaft,
                                            prm.distance_from_point_thrustshaft,
                                            prm.length_thrustshaft,
                                            prm.radius_thrustshaft,
                                            prm.rho_thrustshaft,
                                            prm.mass_inertia_thrustshaft,
                                            3,
                                            dof_handler,
                                            quadrature_formula);
    Shaft engine_shaft = createShaft(prm.useShaft_engineshaft,
                                            prm.distance_from_point_engineshaft,
                                            prm.length_engineshaft,
                                            prm.radius_engineshaft,
                                            prm.rho_engineshaft,
                                            prm.mass_inertia_engineshaft,
                                            4,
                                            dof_handler,
                                            quadrature_formula);
    Shaft propeller_shaft = createShaft(prm.useShaft_propellershaft,
                                            prm.distance_from_point_propellershaft,
                                            prm.length_propellershaft,
                                            prm.radius_propellershaft,
                                            prm.rho_propellershaft,
                                            prm.mass_inertia_propellershaft,
                                            5,
                                            dof_handler,
                                            quadrature_formula);
    //std::vector<Shaft> allShafts = combineShafts(intermediate_shaft, thrust_shaft, engine_shaft, propeller_shaft);
    std::map<int,Shaft> allShafts = combineShaftsToMap(intermediate_shaft, thrust_shaft, engine_shaft, propeller_shaft);

    //const double density_value = 1000; // Replace with your desired density value
    Functions::ConstantFunction<dim> density_function(prm.rho) ;
    double mass_moment_of_inertia = intermediate_shaft.mass_moment_of_inertia;
    pcout <<  "\n        Mass Moment of Inertia: "<< mass_moment_of_inertia <<std::flush;

     //pcout <<  "\n        intermediate Shaft Radius: "<< prm.radius_intermediateshaft <<std::flush;
   
    const FEValuesExtractors::Vector displacement(0);
    typename dealii::DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->subdomain_id() == this_mpi_process)
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;
          cell_res    = 0;

          cell_stiffness_matrix = 0;
          cell_mass_matrix      = 0;
          cell_damping_matrix   = 0;
          cell_displacement = 0;
          cell_velocity = 0;
          cell_accel = 0;
          cell_accel_rotation = 0;
         
          VelocityAngular<dim> velocity_angular(0); //prm.rpm*rpm_to_rps); 
        std::vector<Vector<double>> velocity_angular_values(n_q_points,
                                                                      Vector<double>(dim));
        AccelerationAngular<dim> acceleration_angular(prm.rpm*rpm_to_rps);
        std::vector<Vector<double>> acceleration_angular_values(n_q_points,
                                                                      Vector<double>(dim));
        DisplacementEngine<dim> displacement_engine(prm.rpm*rpm_to_rps,prm.delta_t);
        std::vector<Vector<double>> displacement_engine_values(n_q_points,
                                                                      Vector<double>(dim));
        PropellerForce<dim> propeller_pressure(pressure); 
        std::vector<Vector<double>> propeller_pressure_values(n_q_points,
                                                      Vector<double>(dim));
        Force4Area<dim> propeller_distributed_weight(-prop_weight_pressure,dim-1); 
        std::vector<Vector<double>> propeller_distributed_weight_values(n_q_points,
                                                      Vector<double>(dim));        
        PropellerTorque<dim> propeller_torque(total_torque, polar_moment, (prm.useShaft_propellershaft) ? prm.radius_propellershaft: prm.radius_intermediateshaft) ; 
        std::vector<Vector<double>> propeller_torque_values(n_q_points,
                                                      Vector<double>(dim));

        springbearingLHS<dim> journal_bearing_spring(1,dim-1); 

        std::vector<Vector<double>> journal_bearing_spring_values(n_q_points,
                                                     Vector<double>(dim));  
                                               
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    Point<dim> point = fe_values.quadrature_point(q_point);
                  const SymmetricTensor<2, dim>
                    eps_phi_i = get_strain(fe_values, i, q_point),
                    eps_phi_j = get_strain(fe_values, j, q_point);
 
                  cell_matrix(i, j) += (eps_phi_i *            
                                        stress_strain_tensor_ * 
                                        eps_phi_j              
                                        ) *                    
                                       fe_values.JxW(q_point); 

                  

                  
                  cell_mass_matrix(i, j) +=allShafts[getBoundaryIdForXCoordinateShaft<dim>(allShafts,point(0))].density* (
                                           fe_values.shape_value(i, q_point) *
                                            fe_values.shape_value(j, q_point) *
                                            fe_values.JxW(q_point));
                cell_damping_matrix(i, j) += (prm.mu_rayleigh*cell_mass_matrix(i, j) + prm.lambda_rayleigh*cell_matrix(i, j));
                  cell_stiffness_matrix(i, j) +=  cell_mass_matrix(i, j) +   (1+prm.HHTa)*prm.gamma*prm.delta_t*(cell_damping_matrix(i, j)) + (1+prm.HHTa)*prm.beta*prm.delta_t*prm.delta_t*cell_matrix(i, j);     
                // cell_stiffness_matrix -> N/m * u[m] = F[N], then added stiffness distributed over area: added_stff/area=total_added/Area
                }


          for (unsigned int face = 0; face < n_faces_per_cell; ++face)
            if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() >6))
              {
                fe_face_values.reinit(cell, face);
                journal_bearing_spring.vector_value_list(fe_face_values.get_quadrature_points(), 
                                                                    journal_bearing_spring_values);
                const types::boundary_id boundary_id = cell->face(face)->boundary_id();
                //double boundary_area = std::get<0>(boundary_info[boundary_id]);
                SymmetricTensor<2, dim> rhs_values;
                for (unsigned int q_point = 0; q_point < fe_face_values.n_quadrature_points;++q_point)
                    {
                        
                        Point<dim> point = fe_face_values.quadrature_point(q_point);
                        int boundary_id =  getBoundaryIdForXCoordinateBearing<dim>(allBearings,point(0));
                        Point<dim> min_coordinate (allBearings[boundary_id].end_position,-allBearings[boundary_id].radius_shaft,-allBearings[boundary_id].radius_shaft);
                        Point<dim> max_coordinate (allBearings[boundary_id].start_position,allBearings[boundary_id].radius_shaft,-0.8*allBearings[boundary_id].radius_shaft);
                        if(point_inside_box(point, min_coordinate, max_coordinate))
                        {
                            for (unsigned int i = 0; i < dim; ++i)
                                {
                                    rhs_values[i][i] = std::get<0>(boundaryStiffenessMap[getBoundaryIdForXCoordinateBearing<dim>(allBearings,point[0])])*journal_bearing_spring_values[q_point][i];
                                }
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                                    
                                {
                                    const unsigned int component_i = fe.system_to_component_index(i).first;
                                    const unsigned int component_j = fe.system_to_component_index(j).first;
                                    double temp = fe_face_values[displacement].value(i, q_point)*
                                                rhs_values*fe_face_values[displacement].value(j, q_point)*
                                                fe_face_values.JxW(q_point);
                                    if (isnan(temp))
                                    {
                                    cell_matrix(i, j) +=  0;
                                        }
                                    else
                                    {
                                        cell_matrix(i, j) +=  temp;
                                        cell_damping_matrix(i, j) +=prm.lambda_rayleigh*cell_matrix(i, j);
                                        cell_stiffness_matrix(i, j) +=   (1+prm.HHTa)*prm.gamma*prm.delta_t*(cell_damping_matrix(i, j)) + (1+prm.HHTa)*prm.beta*prm.delta_t*prm.delta_t*cell_matrix(i, j);
                                    }

                                    
                                }
                        }
                    }
              }


           // local_quadrature_points_data is a pointer for PointHistory value when used with *
          


          const PointHistory<dim> *local_quadrature_points_data =
            reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
          body_force.vector_value_list(fe_values.get_quadrature_points(),
                                       body_force_values);
          velocity_angular.vector_value_list(fe_values.get_quadrature_points(), 
                                      velocity_angular_values);
          acceleration_angular.vector_value_list(fe_values.get_quadrature_points(),
                                      acceleration_angular_values);
          displacement_engine.vector_value_list(fe_values.get_quadrature_points(),
                                      displacement_engine_values);
        
          //DynamicIncrementalValues<dim> dynamicincrementalvalues_bearing1(prm.end_time, present_timestep,prm.direction,prm.displacement);
          //std::vector<Vector<double>> propeller_pressure_dofs(n_q_points,
          //                                             Vector<double>(dim));
          std::vector<Vector<double>> dynamicincremental_displacement_bearing1(n_q_points,
                                                        Vector<double>(dim));
          std::vector<Vector<double>> dynamicincremental_velocity_bearing1(n_q_points,
                                                        Vector<double>(dim));          
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                fe.system_to_component_index(i).first;
 
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    
                  const SymmetricTensor<2, dim> &old_stress =
                    local_quadrature_points_data[q_point].old_stress;
                  //for (unsigned int i = 0; i < dim; ++i)
                      cell_res(i) += old_stress * get_strain(fe_values, i, q_point)* fe_values.JxW(q_point);
                    
                         //cell_rhs(i) +=
                         //   (body_force_values[q_point](component_i)*fe_values.shape_value(i, q_point) 
                         //   -old_stress * get_strain(fe_values, i, q_point))
                         //   * fe_values.JxW(q_point);
                        cell_rhs(i) +=body_force_values[q_point](component_i)*fe_values.shape_value(i, q_point)*fe_values.JxW(q_point);//  -old_stress * get_strain(fe_values, i, q_point)* fe_values.JxW(q_point) ;
                    if (timestep_no==1 &&newton_iter==1)
                     {
                        cell_velocity(i)     = fe_values.shape_value(i, q_point) *velocity_angular_values[q_point](component_i)     *fe_values.JxW(q_point);
                        cell_accel(i)        = fe_values.shape_value(i, q_point) *acceleration_angular_values[q_point](component_i) *fe_values.JxW(q_point);
                       //cell_displacement(i) = fe_values.shape_value(i, q_point) *displacement_engine_values[q_point](component_i)  *fe_values.JxW(q_point);
                    }
                    //else
                    //    cell_velocity(i)     = fe_values.shape_value(i, q_point) *velocity_angular_values[q_point](component_i)     *fe_values.JxW(q_point);


                }
            }

          
          
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
            {


              if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() == 1))
              //if (cell->boundary_id() == 1)
                  {
                         
                            fe_face_values.reinit(cell, face);
                            //fe_values.reinit(cell);
                            //if (faces_touched[])
                            propeller_pressure.vector_value_list(fe_face_values.get_quadrature_points(), 
                                                                    propeller_pressure_values);
                                        
                            propeller_torque.vector_value_list(fe_face_values.get_quadrature_points(), 
                                                      propeller_torque_values);
  
                          
                          

                            for (unsigned int q_point = 0; q_point < fe_face_values.n_quadrature_points;
                                ++q_point)
                              {
                         
                                Tensor<1, dim> rhs_values;
                                for (unsigned int i = 0; i < dim; ++i)
                                  {
                                    rhs_values[i] = propeller_torque_values[q_point][i] +propeller_pressure_values[q_point][i];
                                  }
                                for (unsigned int i = 0; i < dofs_per_cell; ++i)            
                                  {
                                        
                                        const unsigned int component_i =
                                                                    fe.system_to_component_index(i).first;
                                        double temp = -fe_face_values[displacement].value(i, q_point)*
                                            rhs_values*
                                            fe_face_values.JxW(q_point);
                                            if (isnan(temp))
                                                {
                                                  cell_rhs(i) +=0;
                                                  if (isnan(fe_face_values.shape_value(i,q_point)))
                                                    count_fe_face_values_nan +=1;
                                                  if (isnan(fe_face_values.JxW(q_point)))
                                                    count_fe_face_JxW_nan+=1;
                                                  if (isnan(propeller_torque_values[q_point](component_i)))
                                                    count_rhs_nan+=1;
                                                }
                                            else
                                              cell_rhs(i) += temp; 
                                       // countboundary1 +=1;


                                  }
                                }
                   }
                  else if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() == 0) && (prm.dynamicmode))
                  {
                           
                    fe_face_values.reinit(cell, face);
                            //fe_values.reinit(cell);
                            //if (faces_touched[])
                            propeller_pressure.vector_value_list(fe_face_values.get_quadrature_points(), 
                                                                    propeller_pressure_values);
                                        
                            propeller_torque.vector_value_list(fe_face_values.get_quadrature_points(), 
                                                      propeller_torque_values);

                            for (unsigned int q_point = 0; q_point < fe_face_values.n_quadrature_points;
                                ++q_point)
                              {
                         
                                Tensor<1, dim> rhs_values;
                                for (unsigned int i = 0; i < dim; ++i)
                                  {
                                    rhs_values[i] = propeller_torque_values[q_point][i] +propeller_pressure_values[q_point][i];
                                  }
                                for (unsigned int i = 0; i < dofs_per_cell; ++i)            
                                  {
                                        
                                        const unsigned int component_i =
                                                                    fe.system_to_component_index(i).first;
                                        double temp =fe_face_values[displacement].value(i, q_point)*
                                            rhs_values*
                                            fe_face_values.JxW(q_point);
                                            if (isnan(temp))
                                                {
                                                  cell_rhs(i) +=0;
                                                  if (isnan(fe_face_values.shape_value(i,q_point)))
                                                    count_fe_face_values_nan +=1;
                                                  if (isnan(fe_face_values.JxW(q_point)))
                                                    count_fe_face_JxW_nan+=1;
                                                  if (isnan(propeller_torque_values[q_point](component_i)))
                                                    count_rhs_nan+=1;
                                                }
                                            else
                                              cell_rhs(i) +=0; // temp; 
                                       // countboundary1 +=1;


                                  }
                                }
                   }
                  else if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() == 3))
                  {
                              
                      propeller_distributed_weight.vector_value_list(fe_face_values.get_quadrature_points(), 
                                                  propeller_distributed_weight_values);

                      for (unsigned int q_point = 0; q_point < fe_face_values.n_quadrature_points;
                          ++q_point)
                        {
                          Tensor<1, dim> rhs_values;
                          for (unsigned int i = 0; i < dim; ++i)
                            {
                              rhs_values[i] = propeller_distributed_weight_values[q_point][i];
                            }
                          for (unsigned int i = 0; i < dofs_per_cell; ++i)            
                            {
                              double temp = -fe_face_values[displacement].value(i, q_point)*
                                            rhs_values*
                                            fe_face_values.JxW(q_point);
                            if (isnan(temp))
                                {
                                  cell_rhs(i) += 0;
                                     }
                            else
                              cell_rhs(i) += temp; 
                            }
                        }
                    }
                    //else if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() >=4) && timestep_no>1)
                    //{
                    //    fe_face_values.reinit(cell, face);
                    //    journal_bearing_spring.vector_value_list(fe_face_values.get_quadrature_points(), 
                    //                                                        journal_bearing_spring_values);
                    //    Tensor<1, dim> rhs_values;
                    //    for (unsigned int q_point = 0; q_point < fe_face_values.n_quadrature_points;++q_point)
                    //        {
                    //            Point<dim> point = fe_face_values.quadrature_point(q_point);
                    //            
                    //            if(point(dim-1)<=max_z_spring && point(dim-1)>=min_z_spring )
                    //            {
                    //                //point_displacement ;
                    //                //if (prm.dynamicmode)
                    //                //   point_displacement= get_point_displacement(dof_handler, displacement_locally_relevant_solution, cell, q_point);
                    //                //else
                    //                  //Point<dim>   point_displacement = get_point_displacement(dof_handler, displacement_n, cell, q_point);
//
                    //                for (unsigned int i = 0; i < dim; ++i)
                    //                    {
                    //                        rhs_values[i] = journal_bearing_spring_values[q_point][i];
                    //                    }
                    //                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    //                    {
                    //                        const unsigned int component_i = fe.system_to_component_index(i).first;
                    //                        double temp = fe_face_values[displacement].value(i, q_point)*
                    //                                    rhs_values*0*
                    //                                    fe_face_values.JxW(q_point);
                    //                        if (isnan(temp))
                    //                            {
                    //                            cell_rhs(i) += 0;
                    //                                }
                    //                        else
                    //                            cell_rhs(i) += temp; 
                    //                    }
                    //            }
//
                    //        }
                    //}
                  //else if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id() >=4 && prm.dynamicmode && !prm.useReynolds )
                  //  {
                  //      
                  //        dynamicincrementalvalues_bearing1.vector_value_list_displacement(fe_values.get_quadrature_points(), 
                  //                                    dynamicincremental_displacement_bearing1);
                  //        dynamicincrementalvalues_bearing1.vector_value_list_velocity(fe_values.get_quadrature_points(), 
                  //                                    dynamicincremental_velocity_bearing1);
                  //                                    
                  //       for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                  //              {
                  //                Tensor<1, dim> vel_rhs_values;
                  //                Tensor<1, dim> disp_rhs_values;
                  //                for (unsigned int i = 0; i < dim; ++i)
                  //                  {
                  //                    vel_rhs_values[i] =  dynamicincremental_velocity_bearing1[q_point](i);
                  //                    disp_rhs_values[i] =  dynamicincremental_displacement_bearing1[q_point](i);
                  //                  }
                  //                for (unsigned int i = 0; i < dofs_per_cell; ++i)            
                  //                  {
                  //                  
                  //                    cell_displacement(i) =  fe_face_values[displacement].value(i, q_point)*disp_rhs_values*fe_face_values.JxW(q_point);
                  //                    cell_velocity(i) = fe_face_values[displacement].value(i, q_point)*vel_rhs_values*fe_face_values.JxW(q_point);
                  //                  }
                  //                }
                  //    }
              }
            //}
              
                        
                  
              


          cell->get_dof_indices(local_dof_indices);
          //std::cout << "For dof =  matrices construvted" << std::endl;
          constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_matrix,
                                                              cell_rhs,
                                                              local_dof_indices,
                                                              system_matrix,
                                                              system_rhs);
                                                                                            

          constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_stiffness_matrix,
                                               local_dof_indices,
                                               system_dynamic_matrix);
          constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               system_mass_matrix);     
          constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_damping_matrix,                                               
                                                local_dof_indices,
                                               system_damping_matrix); 
          
          //constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_mass_matrix,
          //                                      local_dof_indices,
          //                                      system_mass_matrix);
                                          
          //{
          //distribute_local_to_global(cell_displacement,displacement_n);
            constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_displacement,
                                                local_dof_indices,
                                               displacement_n);
           constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_velocity,
                                                local_dof_indices,
                                                velocity_n);
            constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_accel,
                                                local_dof_indices,
                                                acceleration_n_prev);
          

                                    

        }


          pcout << "      Matrices for  " << this_mpi_process <<" process done." <<std::endl;
          pcout << "Count fe_face_values NaN="<< count_fe_face_values_nan << std::endl;  
          pcout << "Count fe_face_JxW NaN="<< count_fe_face_JxW_nan << std::endl;
          pcout << "Count rhs NaN="<< count_rhs_nan << std::endl;


 
    
    const FEValuesExtractors::Scalar          x_component(0);
    const FEValuesExtractors::Scalar          bearing_component(dim-1);

    system_matrix.compress(VectorOperation::add);
    system_mass_matrix.compress(VectorOperation::add);
    

    system_damping_matrix.compress(VectorOperation::add);

   

   
    system_dynamic_matrix.compress(VectorOperation::add);

    displacement_n.compress(VectorOperation::add);
    velocity_n.compress(VectorOperation::add);
    acceleration_n.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
   // if (present_timestep> 1)
      if (prm.useReynolds)
        for (const auto& bearingEntry : allBearings)
          {
            const auto& bearing = bearingEntry.second;
            pcout << "      Boundaries initialised for bearing " <<bearing.boundary_id<<std::endl;
            assemble_bearing_rhs(bearing, bearing.eccentricity,bearing.phi);
            pcout << "        Bearing Boundaries initialised." <<std::endl;
            //PETScWrappers::MPI::Vector tmp_rhs(locally_owned_dofs, mpi_communicator);
            //tmp_rhs = bearing_locally_relevant_solution;
            interpolate_bearing_rhs();



            //tmp_rhs.compress(VectorOperation::add);
            //system_rhs.compress(VectorOperation::add);
            //system_rhs.add(1,tmp_rhs);
           
          }

      //!TO BE SEEN
      if (!prm.dynamicmode && !prm.useNR)
      { 
        

        pcout << "        Residual Linear Calced." <<std::flush; 
        system_rhs.compress(VectorOperation::add);
        calculate_residual_linear();
        std::map<types::global_dof_index, double> boundary_values;


        
        for (const auto& bearingEntry : allBearings)
          {
            const auto& bearing = bearingEntry.second;
            const FEValuesExtractors::Scalar          z_component_1(bearing.displacement_1_direction);
            VectorTools::interpolate_boundary_values(dof_handler,
                                      bearing.boundary_id,
                                       IncrementalBoundaryValues<dim>(bearing.displacement_1_endtime, present_timestep,bearing.displacement_1_direction,bearing.displacement_1_displacement),
                                      boundary_values,
                                      fe.component_mask(z_component_1));
            const FEValuesExtractors::Scalar          z_component_2(bearing.displacement_2_direction);

            VectorTools::interpolate_boundary_values(dof_handler,
                                      bearing.boundary_id,
                                       IncrementalBoundaryValues<dim>(bearing.displacement_2_endtime, present_timestep,bearing.displacement_2_direction,bearing.displacement_2_displacement),
                                      boundary_values,
                                      fe.component_mask(z_component_2));
          }
        MatrixTools::apply_boundary_values(
                  boundary_values, system_matrix, tmp, system_rhs, false);

        locally_relevant_solution = tmp;
         pcout << "        Bearing Boundaries added." <<std::flush; 
      }

    else if (prm.dynamicmode && !prm.useNR)
        {
            //if (timestep_no==1)
            calculate_residual_linear();

//
            predictors();
            assemble_rhs_newton(prm.HHTa);
            

            pcout << "        Residual Linear Calced." <<std::flush; 
            system_rhs.compress(VectorOperation::add);
        }


   
    //pcout << "    Assembiling System for " << this_mpi_process <<" process done." <<std::endl;
  }
  
  template <int dim>
  void TopLevel<dim>::assemble_bearing_rhs(const Bearing bearing, 
                                          const double eccentricity, const double phi)
  {
    
     TimerOutput::Scope t(computing_timer, "Assembling Bearing");
     bearing_locally_relevant_solution = 0;
    //std::vector<types::global_dof_index> face_dof_indices(fe.dofs_per_face);
    ////////////////////////////////
    //const IndexSet boundary_dof_set = DoFTools::extract_boundary_dofs(dof_handler,
    //                                                  boundary_id);
    //std::vector<bool> component_mask(dim);
    //const ComponentMask component_mask();
    std::set<types::boundary_id> boundary_ids;
    const std::vector<types::boundary_id> boundary_ids_vector = dof_handler.get_triangulation().get_boundary_ids();
    const std::set<types::boundary_id> all_boundary_ids(boundary_ids_vector.begin(), boundary_ids_vector.end());
    //const std::set< types::boundary_id > boundary_id;
    //component_mask[0] = true;
    //component_mask[1] = true;
    //component_mask[2] = true;
    // std::vector<dealii::IndexSet> 
    //for (unsigned int i = 0; i < n_mpi_processes; ++i)

    const IndexSet boundary_dof_set = DoFTools::extract_boundary_dofs(dof_handler,ComponentMask(),boundary_ids);
  

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    
  

    FEFaceValues<dim> fe_face_values(fe_values.get_fe(), 
                                            face_quadrature_formula,
                                           update_values | update_gradients | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell ;
    const unsigned int dofs_per_face = fe.n_dofs_per_face();
    typename dealii::DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    
    double disp_increm = bearing.displacement_1_displacement/bearing.displacement_1_endtime*present_timestep;
    double bearing_radius = bearing.radius_bearing;
  


    BearingSurfaceMove<dim> h(bearing_radius, eccentricity, phi,disp_increm,bearing.displacement_1_direction,prm.delta_t);  
    std::vector<Vector<double>> h_values(n_q_points,Vector<double>(1));
    std::vector<Vector<double>> h_dot_values(n_q_points,Vector<double>(1));
   
    VelocityAngular<dim> velocity_angular(prm.rpm*rpm_to_rps); 
    std::vector<Vector<double>> velocity_angular_values(n_q_points,
                                                                  Vector<double>(dim));
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    bearing_system_matrix = 0;
    bearing_rhs = 0;
    const FEValuesExtractors::Vector displacement(0);
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;

          for (unsigned int face = 0; face < n_faces_per_cell; ++face)
          {
              fe_face_values.reinit(cell, face);
              //fe_values.reinit(cell);
              //if (faces_touched[])
              
              if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() == bearing.boundary_id))
                {
                  h.h_vector_value_list(fe_face_values.get_quadrature_points(), 
                                                          h_values);
                  h.h_dot_vector_value_list(fe_face_values.get_quadrature_points(),
                                                      h_dot_values);


                  velocity_angular.vector_value_list(fe_face_values.get_quadrature_points(), 
                                      velocity_angular_values);
                  
                for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                {
                  Tensor<1, dim> rhs_values; // = (eps_phi_i)*velocity_angular_values[q_point];
                  Tensor<1, dim> rhs_h_values;
                  Tensor<1, dim> rhs_h_values_m;

                  Tensor<1, dim> rhs_h_dot_values;

                  for (unsigned int k = 0; k < dim; ++k)
                    {
                          rhs_values[k] = velocity_angular_values[q_point][k];
                          rhs_h_values[k] = rhs_values[k]*h_values[q_point][k];
                          rhs_h_values_m[k]= pow( h_values[q_point][k],3);
                          rhs_h_dot_values[k] = h_dot_values[q_point][k];
                    }
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      const unsigned int component_i =
                            fe.system_to_component_index(i).first;
                      const Tensor<1, dim> 
                            eps_phi_i = get_gradient_face(fe_face_values, i, q_point);

        
                      
                       //rhs_value = eps_phi_i * velocity_angular_values;
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        //for (unsigned int i = 0; i < dim; ++i)
                        //  rhs_h_values_m[i] = h_values[q_point][i];
                       
                        const Tensor<1, dim> 
                          //eps_phi_i = get_strain(fe_values, i, q_point),
                          eps_phi_j = get_gradient_face(fe_face_values, j, q_point);
                        cell_matrix(i, j) += eps_phi_j *rhs_h_values_m[component_i]/(12*bearing.viscocity*1e-12)*((eps_phi_i)) *                    
                                              fe_face_values.JxW(q_point);
                      }
                      
                      cell_rhs(i) += eps_phi_i*rhs_h_values* fe_face_values.JxW(q_point)/2;
                      cell_rhs(i) -= fe_face_values[displacement].value(i, q_point)  *rhs_h_dot_values* fe_face_values.JxW(q_point);
                      //cell_rhs(i) += (fe_values.shape_value(i, q_point) ) *rhs_h_values[i]*rhs_values[i] * fe_values.JxW(q_point)/2;
                      //cell_rhs(i) -= (fe_values.shape_value(i, q_point) ) *rhs_h_dot_values[i]* fe_values.JxW(q_point);
                    }
                }
                
                }
          }
          cell->get_dof_indices(local_dof_indices);
          constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_matrix,
                                                              cell_rhs,
                                                              local_dof_indices,
                                                              bearing_system_matrix,
                                                              bearing_rhs);
      }
    bearing_system_matrix.compress(VectorOperation::add);
    bearing_rhs.compress(VectorOperation::add);

    std::map<types::global_dof_index, double> boundary_values;
    for (const auto &id : all_boundary_ids)
    {
      if (id !=bearing.boundary_id )
      {
        VectorTools::interpolate_boundary_values(dof_handler,
                                              id,
                                          Functions::ZeroFunction<dim>(dim),
                                          boundary_values ); 
      }
    }

    PETScWrappers::MPI::Vector distributed_locally_relevant_solution(
                    locally_owned_dofs, mpi_communicator);                               
    MatrixTools::apply_boundary_values(
      boundary_values, bearing_system_matrix, distributed_locally_relevant_solution, bearing_rhs, true);
    
    pcout << "          Cell Matrices Constructed for bearing with boundary_id:"<< bearing.boundary_id << std::endl;
    pcout << "              Bearing norm of rhs is " << bearing_rhs.l2_norm() << std::endl;
    
      //TimerOutput::Scope t(computing_timer, "Solving Bearing");

      distributed_locally_relevant_solution = 0;
      
      //PETScWrappers::PreconditionBoomerAMG preconditioner;
      //data.n_sweeps_coarse = true;
      //PETScWrappers::PreconditionSSOR::AdditionalData data
      PETScWrappers::PreconditionBoomerAMG preconditioner; //(bearing_system_matrix);
      PETScWrappers::PreconditionBoomerAMG::AdditionalData data;      //
      //data.n_sweeps_coarse = true;
      //data.symmetric_operator = true;
      //data.tol = prm.bearing_tol*0.0001;

      //preconditioner
      preconditioner.initialize(bearing_system_matrix, data);   
      
      //PETScWrappers::PreconditionBlockJacobi preconditioner(bearing_system_matrix);
      SolverControl solver_control(20000, std::max({prm.bearing_tol_journal,prm.bearing_tol_engine,prm.bearing_tol_sterntube,prm.bearing_tol_tailtube})*bearing_rhs.l2_norm());
      
      PETScWrappers::SolverCG cg(solver_control, mpi_communicator);
      
      cg.solve(bearing_system_matrix, distributed_locally_relevant_solution, bearing_rhs, preconditioner);
      pcout << "          Bearing System Solved for bearing:"<< bearing.boundary_id <<" in "<<solver_control.last_step()<<" iterations"<< std::endl;
      //constraints_dirichlet_and_hanging_nodes.distribute(distributed_locally_relevant_solution);
      
      bearing_locally_relevant_solution = distributed_locally_relevant_solution;
      //bearing_locally_relevant_solution*=1;
      total_bearings_locally_relevant_solution.add(1,bearing_locally_relevant_solution);
    
          pcout << "          Pressure Norm for bearing:"<< bearing.boundary_id <<" is "<<bearing_locally_relevant_solution.l2_norm()<< std::endl;


       

  }

  template <int dim>
  void TopLevel<dim>::interpolate_bearing_rhs()
  {
    
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    Vector<double> cell_tmp(dofs_per_cell);
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell ;

    //std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    typename dealii::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    Vector<double> tmp;
    tmp.reinit(dof_handler.n_dofs());
    PETScWrappers::MPI::Vector tmp_rhs(
                    locally_owned_dofs, mpi_communicator);
    tmp = bearing_locally_relevant_solution;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    const FEValuesExtractors::Vector displacement(0);
    for (; cell != endc; ++cell)
      if (cell->subdomain_id() == this_mpi_process)
      {
      //for (unsigned int face = 0; face < n_faces_per_cell; ++face)
       for (const auto v : cell->vertex_indices())
         {
           
            fe_values.reinit(cell);
           
            cell_tmp = 0;
            if (vertex_touched[cell->vertex_index(v)] == false)
              {
                vertex_touched[cell->vertex_index(v)] = true;
                 Tensor<1, dim> vertex_displacement;
                for (unsigned int q_point = 0; q_point < n_faces_per_cell; ++q_point)
                 {
                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      vertex_displacement[d] = tmp(cell->vertex_dof_index(v, d));
                      //if (vertex_displacement[d]>0)
                      //    vertex_displacement[d]=0;
                    }
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    
                          //const double pressure_value = tmp[local_dof_indices[i]];
                    cell_tmp(i) +=fe_values[displacement].value(i, q_point)* vertex_displacement* fe_values.JxW(q_point);
                
                  }
                 }
              } 
          }

             cell->get_dof_indices(local_dof_indices);
            constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_tmp,
                                                local_dof_indices,
                                                tmp_rhs);
      }
    system_rhs.compress(VectorOperation::add);
    tmp_rhs.compress(VectorOperation::add);
    system_rhs.add(1,tmp_rhs);
    //
  }


  template <int dim>
  void TopLevel<dim>::predictors()
  { 
    TimerOutput::Scope t(computing_timer, "predictors");
    // Update velocity and acceleration with Newmark-beta method PREDICTORS

    pcout << " Predictors calculations " << std::endl;    
    displacement_n.compress(VectorOperation::add);
    velocity_n.compress(VectorOperation::add);
    acceleration_n.compress(VectorOperation::add);
    //if (timestep_no>1)
    acceleration_n_prev =acceleration_n ;
    //acceleration_n =0;
    acceleration_n.compress(VectorOperation::add);
    velocity_n_prev.compress(VectorOperation::add);
    acceleration_n_prev.compress(VectorOperation::add);
    displacement_n_prev = displacement_n;
    velocity_n_prev = velocity_n;
    //displacement_n  = 0;

    //velocity_n = 0;
    //displacement_n.add(prm.delta_t,velocity_n_prev);
    //displacement_n.add((1/2-prm.beta)*prm.delta_t*prm.delta_t,acceleration_n_prev);
    //displacement_n.add((prm.beta)*prm.delta_t*prm.delta_t,acceleration_n);
//
    //velocity_n.add((1-prm.gamma)*prm.delta_t,acceleration_n_prev);
    //velocity_n.add((prm.gamma)*prm.delta_t,acceleration_n);
    displacement_n.compress(VectorOperation::add);
    velocity_n.compress(VectorOperation::add);
    acceleration_n.compress(VectorOperation::add);
    velocity_n_prev.compress(VectorOperation::add);
    acceleration_n_prev.compress(VectorOperation::add);
  

    //constraints_dirichlet_and_hanging_nodes.distribute(displacement_n);
    //constraints_dirichlet_and_hanging_nodes.distribute(velocity_n);
    //constraints_dirichlet_and_hanging_nodes.distribute(acceleration_n);
  }
  template <int dim>
  void TopLevel<dim>::time_stepping(const PETScWrappers::MPI::Vector &du)
  {
    TimerOutput::Scope t(computing_timer, "time_stepping");
   // acceleration_n = distrivute
   //parallel::distributed::
   //SolutionTransfer<dim,VectorType,DoFHandler<dim,spacedim>>
   // sol_trans(dof_handler);
   // sol_trans.deserialize(distributed_vector);
    PETScWrappers::MPI::Vector du_tmp;
    du_tmp.reinit(locally_owned_dofs, mpi_communicator);
    du_tmp = du;
    du_tmp.compress(VectorOperation::add);
    acceleration_n.add(1,du);
    acceleration_n.compress(VectorOperation::add);

    pcout << " Timestepping calculations " << std::endl;       
        

    //acceleration_n.distribute_loacl_to_global(locally_relevant_solution);

    // Update velocity and acceleration with Newmark-beta method CORRECTORS for the next time-step
    velocity_n.add((prm.gamma)*prm.delta_t,du);
    displacement_n.add(prm.beta*prm.delta_t*prm.delta_t,du);
    acceleration_n.compress(VectorOperation::add);
    velocity_n.compress(VectorOperation::add);
    displacement_n.compress(VectorOperation::add);

//
    //locally_relevant_solution = displacement_n;
    
    //constraints_dirichlet_and_hanging_nodes.distribute(displacement_n);
    //constraints_dirichlet_and_hanging_nodes.distribute(velocity_n);
    //constraints_dirichlet_and_hanging_nodes.distribute(acceleration_n);
     if (!prm.useNR)
     {
            acceleration_n_prev = acceleration_n;
            velocity_n_prev = velocity_n;
            displacement_n_prev = displacement_n;
            system_rhs_prev = system_rhs;
            displacement_n_prev.compress(VectorOperation::add);
            velocity_n_prev.compress(VectorOperation::add);
            acceleration_n_prev.compress(VectorOperation::add);
            system_rhs_prev.compress(VectorOperation::add);
     }   

  }


  template <int dim>
  void TopLevel<dim>::assemble_rhs_newton(const double a_HHT)
  {
    ///////
  
    
    PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
    tmp.reinit(locally_owned_dofs, mpi_communicator);
    PETScWrappers::MPI::Vector tmp_2(locally_owned_dofs, mpi_communicator);
    tmp_2.reinit(locally_owned_dofs, mpi_communicator);
    //{
    //      pcout << "      Predictors done." <<std::endl;
    //      predictors();
    //}
    system_matrix.compress(VectorOperation::add);
    system_mass_matrix.compress(VectorOperation::add);
    system_damping_matrix.compress(VectorOperation::add);
    system_dynamic_matrix.compress(VectorOperation::add);

    displacement_n.compress(VectorOperation::add);
    velocity_n.compress(VectorOperation::add);
    acceleration_n.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    displacement_n_prev.compress(VectorOperation::add);
    velocity_n_prev.compress(VectorOperation::add);
    acceleration_n_prev.compress(VectorOperation::add);
    system_rhs_prev.compress(VectorOperation::add);
    //pcout << "      Newton Matrices Compressed." <<std::endl; 
    newton_rhs.reinit(locally_owned_dofs, mpi_communicator);
     std::map<types::global_dof_index, double> boundary_values;

    //VectorTools::interpolate_boundary_values(dof_handler,
    //                          0,
    //                          DisplacementEngine<dim>(prm.rpm*rpm_to_rps,prm.delta_t),
    //                          boundary_values);
    //MatrixTools::apply_boundary_values(
    //      boundary_values, system_matrix, tmp, displacement_n, false);
    newton_rhs = 0;
    newton_rhs.add((1+a_HHT),system_rhs);
   // pcout << "      Vector system_rhs Calced." <<std::endl; 

    newton_rhs.add(-a_HHT,system_rhs_prev);
    //pcout << "      Vector system_rhs_prev Calced." <<std::endl; 

   // pcout << "      Vector Matrices Calced." <<std::endl; 
    tmp.reinit(locally_owned_dofs, mpi_communicator);
    system_damping_matrix.vmult(tmp,velocity_n);
    
    newton_rhs.add(-(1+a_HHT),tmp);
    tmp.reinit(locally_owned_dofs, mpi_communicator);
    system_damping_matrix.vmult(tmp,velocity_n_prev);
    newton_rhs.add(a_HHT,tmp);

    system_damping_matrix.vmult(tmp,acceleration_n_prev);
    newton_rhs.add(a_HHT*prm.delta_t*prm.gamma,tmp);
    system_matrix.vmult(tmp,acceleration_n_prev);
    newton_rhs.add(a_HHT*prm.delta_t*prm.delta_t*prm.beta,tmp);
    std::map<types::global_dof_index, double> boundary_values_disp;
    VectorTools::interpolate_boundary_values(dof_handler,
                                        0,
                                      Functions::ZeroFunction<dim>(dim),
                                      boundary_values_disp);
            MatrixTools::apply_boundary_values(
                  boundary_values_disp, system_dynamic_matrix, acceleration_n, newton_rhs, false);    
    //pcout << "      Matrices Matrices Calced." <<std::endl; 
    newton_rhs.compress(VectorOperation::add);

    
    pcout << "      Newton Matrices Assembled." <<std::endl; 

  }
  
  template <int dim>
  void TopLevel<dim>::calculate_residual_linear()
  {
     FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    const unsigned int  dofs_per_cell= fe.n_dofs_per_cell();
    Vector<double>     cell_res(dofs_per_cell);
    PETScWrappers::MPI::Vector newton_res;
    newton_res.reinit(locally_owned_dofs, mpi_communicator);
    newton_res = 0;

    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    typename dealii::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->subdomain_id() == this_mpi_process)
        {
          cell_res = 0;
          fe_values.reinit(cell);
          const PointHistory<dim> *local_quadrature_points_data =
            reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
          
          for (unsigned int q_point = 0; q_point < quadrature_formula.size(); ++q_point)
            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              cell_res(i) += local_quadrature_points_data[q_point].old_stress 
              * get_strain(fe_values, i, q_point)* fe_values.JxW(q_point);     

          cell->get_dof_indices(local_dof_indices);  

        
    
           constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_res,
                                                local_dof_indices,
                                                newton_res);
        }
    newton_res.compress(VectorOperation::add);
    system_rhs.add(-1,newton_res);
      
  }

  template <int dim>
  void TopLevel<dim>::newton()
  {
    TimerOutput::Scope t(computing_timer, "assemble_newton");
    // Update velocity and acceleration with Newmark-beta method PREDICTORS

   // pcout << " Newton calculations " << std::endl;    
    double residual_norm;
    double previous_residual_norm = -std::numeric_limits<double>::max();

    double disp_norm,
           previous_disp_norm = 0;

    const unsigned int max_newton_iter = prm.NRmaxiter;

    for (unsigned int newton_step = 1; newton_step <= max_newton_iter; ++newton_step)
      {
        total_bearings_locally_relevant_solution = 0;
        displacement_locally_relevant_solution = 0;
        locally_relevant_solution=0;
        pcout  << "  Newton Iteration: " << newton_step << std::endl;
        newton_iter = newton_step;
        if (newton_step == 1)
            {
              if ( timestep_no == 1)
                {
                  acceleration_n_prev = 0;
                  velocity_n_prev = 0;
                  displacement_n_prev = 0;
                  
                  system_rhs_prev = 0;
                }
               
              pcout << "        Assembling system..." <<  std::endl;
              
             
              if ( timestep_no == 1)
              {

                assemble_system();
                //assemble_rhs_newton(prm.HHTa);

                acceleration_n_prev = acceleration_n;
                velocity_n_prev = velocity_n;
                displacement_n_prev = displacement_n;
               
                system_rhs_prev = system_rhs;
                //acceleration_n=0;
                
              }
              predictors();
              assemble_rhs_newton(prm.HHTa);
              
              
            }
      
        

        
        pcout << "        norm of rhs is " << newton_rhs.l2_norm() << std::endl;
         const unsigned int n_iterations = solve_linear_problem();
        pcout << "        Solver converged in " << n_iterations << " iterations."
          << std::endl;
          if (prm.dynamicmode)
              move_mesh(displacement_locally_relevant_solution);
          else
              move_mesh(locally_relevant_solution);
        pcout << "      Updating quadrature point data..." <<  std::endl;
        {
          update_quadrature_point_history();
        }
        {
          TimerOutput::Scope t(computing_timer, "output_results");
          

          output_results();
         
        }
        //
        assemble_system();
        assemble_rhs_newton(prm.HHTa);
        //acceleration_n_prev = acceleration_n;
        //velocity_n_prev = velocity_n;
        //displacement_n_prev = displacement_n;
 
        //system_rhs_prev = system_rhs; 
       
        if (newton_step == 1)
        {
          previous_disp_norm = locally_relevant_solution.l2_norm();
          disp_norm = locally_relevant_solution.l2_norm();
          pcout << "      Displacement norm is " << disp_norm << std::endl;
 
        }
      
          if (newton_rhs.l2_norm()<=prm.NRtol)
            {
              pcout << "      Newton Iteration: " << newton_step << " converged." << std::endl;
              break;
              
            }

          if (disp_norm < previous_disp_norm)
          {
            pcout << "      Displacement norm is " << disp_norm << std::endl;
            pcout << "      Residual norm is " << residual_norm << std::endl;
            pcout << "      Newton Iteration: " << newton_step << " converged." << std::endl;
            break;
          }
          else
          {
            pcout << "      Displacement norm is " << disp_norm << std::endl;

            pcout << "      Newton Iteration: " << newton_step << " not converged." << std::endl;
          }
          previous_residual_norm = residual_norm;
          previous_disp_norm = disp_norm;
        
  
      }
      system_matrix.compress(VectorOperation::add);
      system_mass_matrix.compress(VectorOperation::add);
      system_damping_matrix.compress(VectorOperation::add);
      system_dynamic_matrix.compress(VectorOperation::add);

      displacement_n.compress(VectorOperation::add);
      velocity_n.compress(VectorOperation::add);
      acceleration_n.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
      displacement_n_prev.compress(VectorOperation::add);
      velocity_n_prev.compress(VectorOperation::add);
      acceleration_n_prev.compress(VectorOperation::add);
      system_rhs_prev.compress(VectorOperation::add);

      acceleration_n_prev = acceleration_n;
      velocity_n_prev = velocity_n;
      displacement_n_prev = displacement_n;
      system_rhs_prev = system_rhs;
      displacement_n_prev.compress(VectorOperation::add);
      velocity_n_prev.compress(VectorOperation::add);
      acceleration_n_prev.compress(VectorOperation::add);
      system_rhs_prev.compress(VectorOperation::add);
 
     
  

  }

  template <int dim>
  void TopLevel<dim>::solve_timestep()
  {
    //TimerOutput::Scope t(computing_timer, "solve_timestep");

    pcout << "      Assembling system..." << std::flush;
    assemble_system();
    if (!prm.dynamicmode)
        pcout << "      norm of rhs is " << system_rhs.l2_norm() << std::endl;
    else
        pcout << "      norm of rhs is " << newton_rhs.l2_norm() << std::endl;
    //if (prm.dynamicmode)
    //    pcout << "      norm of rhs is " << system_dynamic_matrix.l1_norm() << std::endl;
    //else
    //    pcout << "      norm of rhs is " << system_matrix.l1_norm() << std::endl;

    
    const unsigned int n_iterations = solve_linear_problem();
 
    pcout << "      Solver converged in " << n_iterations << " iterations."
          << std::endl;
 
    pcout << "      Updating quadrature point data..." << std::flush;
    update_quadrature_point_history();
    pcout << std::endl;
  }

  template <int dim>
  void TopLevel<dim>::solve_timestep_newton()
  {
    //TimerOutput::Scope t(computing_timer, "solve_timestep");
    pcout << "Running Newton Method..." << std::flush;
    

    newton();

 
 
 
   
    pcout << std::endl;
  }
 
 
 
  template <int dim>
  unsigned int TopLevel<dim>::solve_linear_problem()
  {
    TimerOutput::Scope t(computing_timer, "solve_problem");
    //
   
    PETScWrappers::MPI::Vector distributed_locally_relevant_solution(
                  locally_owned_dofs, mpi_communicator);
    //hanging_node_constraints.set_zero(distributed_locally_relevant_solution);
    //hanging_node_constraints.set_zero(system_rhs);

          
    SolverControl solver_control(prm.max_iter,
                                 prm.tol * system_rhs.l2_norm());
 

    if (prm.dynamicmode)
      {
        distributed_locally_relevant_solution = acceleration_n;
        //PETScWrappers::SolverCG cg(solver_control, mpi_communicator);
        PETScWrappers::SolverCG cg(solver_control, mpi_communicator);
        //PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
        //PETScWrappers::PreconditionBoomerAMG preconditioner; //(system_matrix);
        //  
        //  
        ////data.n_sweeps_coarse = true;
        ////   //data.symmetric_operator = true;
        ////    //data.aggressive_coarsening_num_levels = 4;
        //preconditioner.initialize(system_dynamic_matrix, data);
        PETScWrappers::PreconditionBlockJacobi preconditioner(system_dynamic_matrix);
        if (!prm.useNR) 
        {
            cg.solve(system_dynamic_matrix,
                    distributed_locally_relevant_solution,
                    newton_rhs,
                    preconditioner);
        }
        else
        {
            cg.solve(system_dynamic_matrix,
                    distributed_locally_relevant_solution,
                    newton_rhs,
                    preconditioner);
        }
            //acceleration_n = distributed_locally_relevant_solution;
            
            //constraints_dirichlet_and_hanging_nodes.distribute(distributed_locally_relevant_solution);
            locally_relevant_solution = distributed_locally_relevant_solution;
            pcout << "    Timestepping " << std::endl;

          time_stepping(locally_relevant_solution);
      
          displacement_locally_relevant_solution = displacement_n;
         // constraints_dirichlet_and_hanging_nodes.distribute(distributed_locally_relevant_solution);
          displacement_locally_relevant_solution.compress(VectorOperation::add);
      
          total_displacement.add(1.0,displacement_n);
            total_displacement.compress(VectorOperation::add);
          //total_displacement.compress(VectorOperation::add);
      }
    else
    {

          //PETScWrappers::MPI::Vector distributed_locally_relevant_solution(locally_owned_dofs, mpi_communicator);

          PETScWrappers::SolverCG cg(solver_control, mpi_communicator);
          distributed_locally_relevant_solution = locally_relevant_solution;
          //PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
          PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
          //preconditioner.initialize(system_matrix, data);
          
          cg.solve(system_matrix,
             distributed_locally_relevant_solution,
             system_rhs,
             preconditioner);
          //solver.solve(system_matrix,
          //       distributed_locally_relevant_solution,
          //       system_rhs,
          //       preconditioner);
         
          constraints_dirichlet_and_hanging_nodes.distribute(distributed_locally_relevant_solution);
          locally_relevant_solution = distributed_locally_relevant_solution;
        
            total_displacement.add(1.0,locally_relevant_solution);
           total_displacement.compress(VectorOperation::add);

    }

    

 
    return solver_control.last_step();
  }

  template <int dim>
  void TopLevel<dim>::solve_modes(std::string solver_name, std::string preconditioner_name)
  {
    assemble_system();

    TimerOutput::Scope t(computing_timer, "solve_modes");
    
    //double min_spurious_eigenvalue = std::numeric_limits<double>::max(),
    //        max_spurious_eigenvalue = -std::numeric_limits<double>::max();
//
    //for (unsigned int i = 0; i < locally_owned_dofs.n_elements(); ++i)
    //  if (constraints.is_constrained(i))
    //    {
    //      const double ev         = system_matrix(i, i) / system_mass_matrix(i, i);
    //      min_spurious_eigenvalue = std::min(min_spurious_eigenvalue, ev);
    //      max_spurious_eigenvalue = std::max(max_spurious_eigenvalue, ev);
    //    }
    ////if (this_mpi_process==0)
    //  std::cout << "   Spurious eigenvalues are all in the interval " << '['
    //            << min_spurious_eigenvalue << ',' << max_spurious_eigenvalue
    //            << ']' << std::endl;



   // if (this_mpi_process==0)
     pcout <<"EigenValues Calculation" <<std::endl;

    

    PETScWrappers::PreconditionBase *preconditioner = nullptr;

    deallog << preconditioner_name << std::endl;
    if (preconditioner_name == "Jacobi")
      {
        preconditioner =
          new PETScWrappers::PreconditionJacobi(mpi_communicator);
      }
    else if (preconditioner_name == "Boomer")
      {
        PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
        data.symmetric_operator = true;

        preconditioner =
          new PETScWrappers::PreconditionBoomerAMG(mpi_communicator, data);
      }
    else if (preconditioner_name == "BlockJacobi")
      {
        preconditioner =
          new PETScWrappers::PreconditionBlockJacobi(mpi_communicator);
      }
    else
      {
        AssertThrow(false, ExcMessage("Unsupported preconditioner"));
      }

    SolverControl   linear_solver_control(dof_handler.n_dofs(),
                                                1e-16,
                                                /*log_history*/ false,
                                                /*log_results*/ false);
    PETScWrappers::SolverCG linear_solver(linear_solver_control);
    linear_solver.initialize(*preconditioner);

    SolverControl solver_control(prm.max_iter,
                                  1e-16,
                                  /*log_history*/ false,
                                  /*log_results*/ false);

    SLEPcWrappers::SolverBase *eigensolver;

    deallog << solver_name << std::endl;
    // Get a handle on the wanted eigenspectrum solver
    if (solver_name == "KrylovSchur")
      {
        eigensolver =
          new dealii::SLEPcWrappers::SolverKrylovSchur(solver_control,
                                                       mpi_communicator);
      }

    else if (solver_name == "GeneralizedDavidson")
      {
        eigensolver = new dealii::SLEPcWrappers::SolverGeneralizedDavidson(
          solver_control, mpi_communicator);
      }
    else if (solver_name == "JacobiDavidson")
      {
        eigensolver =
          new dealii::SLEPcWrappers::SolverJacobiDavidson(solver_control,
                                                          mpi_communicator);
      }
    else if (solver_name == "Lanczos")
      {
        eigensolver =
          new dealii::SLEPcWrappers::SolverLanczos(solver_control,
                                                   mpi_communicator);
      }
    else if (solver_name == "Arnoldi")
      {
        eigensolver =
          new dealii::SLEPcWrappers::SolverArnoldi(solver_control,
                                                   mpi_communicator);
      }
    else
      {
        AssertThrow(false, ExcMessage("not supported eigensolver"));

        eigensolver =
          new dealii::SLEPcWrappers::SolverKrylovSchur(solver_control,
                                                       mpi_communicator);
      }


    // Set the initial vector. This is optional, if not done the initial vector
    // is set to random values
    //    PETScWrappers::set_option_value("-pc_factor_mat_solver_type",
    //                               "mumps");
    SLEPcWrappers::TransformationShiftInvert::AdditionalData additional_data(
          0);
    SLEPcWrappers::TransformationShiftInvert spectral_transformation(mpi_communicator, additional_data);
   ////// SLEPcWrappers::TransformationShiftInvert::set_matrix_mode()
    spectral_transformation.set_solver(linear_solver);
////
    eigensolver->set_transformation(spectral_transformation);
    // Set the initial vector. This is optional, if not done the initial vector
    // is set to random values
   //     PETScWrappers::set_option_value("-st_pc_factor_mat_solver_package", "mumps");
   //eigensolver->set_initial_space(eigenfunctions);
   eigensolver->set_which_eigenpairs(EPS_TARGET_MAGNITUDE); ///
   //eigensolver->
   eigensolver->set_problem_type(EPS_GHEP);

    //PETScWrappers::set_option_value("-pc_factor_mat_solver_type",
    //                                "mumps");

    // Set the initial vector. This is optional, if not done the initial vector
    // is set to random values
    
    eigensolver->solve(system_matrix,
                       system_mass_matrix,
                       eigenvalues,
                       eigenfunctions,
                       eigenfunctions.size());

    //eigensolver->set_initial_space(eigenfunctions);
//
//
    //eigensolver->set_which_eigenpairs(EPS_LARGEST_REAL);
    //eigensolver->set_problem_type(EPS_GHEP);
//
    //eigensolver->solve(system_matrix,
    //                   system_mass_matrix,
    //                   eigenvalues,
    //                   eigenfunctions,
    //                   eigenfunctions.size());
    //eigensolver->set_initial_space(eigenfunctions);
    
     
    const double               precision = 1e-5;
    PETScWrappers::MPI::Vector Ax(eigenfunctions[0]), Bx(eigenfunctions[0]);
    for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
      {
        system_mass_matrix.vmult(Bx, eigenfunctions[i]);

        for (unsigned int j = 0; j < eigenfunctions.size(); ++j)
          Assert(std::abs(eigenfunctions[j] * Bx - (i == j)) < precision,
                  ExcMessage("Eigenvectors " + Utilities::int_to_string(i) +
                            " and " + Utilities::int_to_string(j) +
                            " are not orthonormal!"));

        system_matrix.vmult(Ax, eigenfunctions[i]);
        Ax.add(-1.0 * eigenvalues[i], Bx);
        Assert(Ax.l2_norm() < precision,
                ExcMessage(Utilities::to_string(Ax.l2_norm())));
      }
    


    for (auto &eigenfunction : eigenfunctions)
        eigenfunction /= eigenfunction.linfty_norm();
  //
//
    delete preconditioner;
    delete eigensolver;
    ////std::cout << std::endl;
    if (this_mpi_process==0)
      for (unsigned int i = 0; i < eigenvalues.size(); ++i)
        std::cout << "      Eigenvalue " << i << " : " << eigenvalues[i]
                  << std::endl;
   // for (auto &eigenfunction : eigenfunctions)
   //   eigenfunction /= eigenfunction.linfty_norm();
   // return solver_control.last_step();
   // DataOut<dim> data_out_eigen;
 //
   // data_out_eigen.attach_dof_handler(dof_handler);
 //
   // for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
   //         data_out_eigen.add_data_vector(eigenfunctions[i],
   //                            std::string("eigenfunction_") +
   //                              Utilities::int_to_string(i));
  //
   // 
 //
   // data_out_eigen.build_patches();
 //
   // std::ofstream output("eigenvectors.vtk");
   // data_out_eigen.write_vtk(output);

  }
 template <int dim>
  void TopLevel<dim>::solve_modes_serial()
  {
    assemble_system();
    //PETScWrappers::SparseMatrix             stiffness_matrix, mass_matrix;
    //stiffness_matrix = system_matrix;
    //mass_matrix = system_mass_matrix;
    
    
  }
 
  template <int dim>
  void TopLevel<dim>::output_results() const
  {
    if (this_mpi_process==0)
    {
      std::filesystem::path dir = "./"+prm.outputfolder;

      // Check if directory does not exist and create it
      if (!std::filesystem::exists(dir))
      {
          std::filesystem::create_directory(dir);
          pcout << "Directory created\n"<<std::flush;
      }
      else
      {
          pcout << "Directory already exists\n"<<std::flush;
      }
    }
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    Vector<double> cell_tmp(dofs_per_cell);
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell ;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    
    std::vector<std::string> solution_names;
    switch (dim)
      {
        case 1:
          solution_names.emplace_back("delta_x");
          break;
        case 2:
          solution_names.emplace_back("delta_x");
          solution_names.emplace_back("delta_y");
          break;
        case 3:
          solution_names.emplace_back("delta_x");
          solution_names.emplace_back("delta_y");
          solution_names.emplace_back("delta_z");
          
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    if (prm.dynamicmode)
      {
        //data_out.add_data_vector(displacement_n, solution_names);
        data_out.add_data_vector(acceleration_n, "acceleration");
        data_out.add_data_vector(velocity_n,"velocity");
        data_out.add_data_vector(displacement_n, "displacement");
        data_out.add_data_vector(displacement_locally_relevant_solution, solution_names);
     
      }

    else
      data_out.add_data_vector(locally_relevant_solution, solution_names);  

    if (timestep_no==1)
    {
        DataPostprocessors::BoundaryIds<dim> boundary_ids;
        DataOutFaces<dim> data_out_face;
        FE_Q<dim>         dummy_fe(1);
        
        DoFHandler<dim>   dummy_dof_handler(triangulation);
        

        
        dummy_dof_handler.distribute_dofs(dummy_fe);
    
        Vector<double> dummy_solution (dummy_dof_handler.n_dofs());
        //std::cout << "Dummy Solution dofs "<<dummy_dof_handler.n_dofs()<<  std::endl;

        data_out_face.attach_dof_handler(dummy_dof_handler);
        data_out_face.add_data_vector(dummy_solution, boundary_ids);
        
        data_out_face.build_patches();
        data_out_face.write_vtu_in_parallel(
                  prm.outputfolder+"/boundary_id.vtu",mpi_communicator);
    }
    //double max_stresses = -1e+20;
    //double max_x_coordinate = -1e+20;
    //std::vector<Vector<double>> center(triangulation.n_active_cells(),Vector<double>(dim));
    Vector<double> centerx(triangulation.n_active_cells());
    Vector<double> centery(triangulation.n_active_cells());
    Vector<double> centerz(triangulation.n_active_cells());
   // total_displacement.compress(VectorOperation::add);
    //if (prm.dynamicmode)
    //{
    //    total_displacement.add(1.0,displacement_n);
    //     //total_displacement.compress(VectorOperation::add);
    //}
    //else
    //{
    //    total_displacement.add(1.0,locally_relevant_solution);
    //     //total_displacement.compress(VectorOperation::add);
    //}
    //total_displacement.compress(VectorOperation::add);

    //data_out.add_data_vector(total_displacement, "total_displacement");
    //data_out.add_data_vector(total_displacement_y, "total_displacement_y");
    //data_out.add_data_vector(total_displacement_z, "total_displacement_z");
    //total_displacement_phi.add(1,)
    Vector<double> norm_of_stress(triangulation.n_active_cells());
    Vector<double> pressure_bearings(triangulation.n_active_cells());
     Vector<double> max_stress(triangulation.n_active_cells());
   
    Vector<double> von_mises_stress(triangulation.n_active_cells());
    //std::vector<Vector<double>> total_disp_vector(triangulation.n_active_cells(),Vector<double>(dim));

    //std::cout << "Face quardature Formula size=" << GeometryInfo<dim>::faces_per_cell << " with points per cell=" << quadrature_formula.size() << std::endl;
    
    std::vector<Vector<double>> face_index(triangulation.n_active_cells(),Vector<double>(GeometryInfo<dim>::faces_per_cell));
    //Vector<double> max_stress(triangulation.n_active_cells());
  
    Vector<double> tmp;
    SymmetricTensor<2, dim> stress_at_qpoint;

    tmp.reinit(dof_handler.n_dofs());
    Vector<double> delta_total;
    delta_total.reinit(triangulation.n_active_cells());
    PETScWrappers::MPI::Vector tmp_rhs(
                    locally_owned_dofs, mpi_communicator);
    tmp = total_bearings_locally_relevant_solution;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    typename dealii::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

      //Vector<double>  VM_stress_field (dof_handler.n_dofs()),
      //     local_VM_stress_values_at_qpoints (quadrature_formula.size()),
      //     local_VM_stress_fe_values (fe.dofs_per_cell);
    
    const FEValuesExtractors::Vector displacement(0);
      //unsigned int count_faces=0;
    for (; cell != endc; ++cell)
      if (cell->subdomain_id() == this_mpi_process)
          {
            // unsigned int count_faces_internal=0;
            const Point<dim> cell_center = cell->center();
            
            centerx(cell->active_cell_index())= cell_center(0);
            if (dim == 2)
              centery(cell->active_cell_index()) = cell_center(1);
            if (dim == 3)
            {
              centery(cell->active_cell_index()) = cell_center(1);
              centerz(cell->active_cell_index()) = cell_center(2);

            }
            fe_values.reinit(cell);
            //center[cell->active_cell_index()](1) = cell_center(1);
            //if (dim == 3)
           //   center[cell->active_cell_index()](2) = cell_center(2);

            SymmetricTensor<2, dim> accumulated_stress;
            double local_VM_stress_values_at_qpoints;
             double VM_stress_max (0);

            Tensor<1, dim> accumulated_pressure;
            for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
              {
                stress_at_qpoint = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer())[q]
                  .old_stress;
                accumulated_stress +=
                reinterpret_cast<PointHistory<dim> *>(cell->user_pointer())[q]
                  .old_stress;
                if (VM_stress_max<stress_at_qpoint.norm())
                  VM_stress_max=stress_at_qpoint.norm();
                //for (unsigned int i=0; i<dim; ++i)
                //  for (unsigned int j=i; j<dim; ++j)
                //    {
                //      local_history_stress_values_at_qpoints[i][j](q) = stress_at_qpoint[i][j];
                //    }
                local_VM_stress_values_at_qpoints = get_von_Mises_stress(stress_at_qpoint);


              }
            
          

              //accumulated_pressure +=total_bearings_locally_relevant_solution(cell->user_pointer())[q];
            
    
            for (const auto v : cell->vertex_indices())
              {
                  cell->get_dof_indices(local_dof_indices);
                
                
                  cell_tmp = 0;
                  if (vertex_touched[cell->vertex_index(v)] == false)
                    {
                      vertex_touched[cell->vertex_index(v)] = true;
                      Tensor<1, dim> vertex_displacement;
                      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                      {
                        for (unsigned int d = 0; d < dim; ++d)
                          {             
                            //vertex_displacement[d] = displacement_n(cell->vertex_dof_index(v, d));
                            accumulated_pressure[d] +=tmp(cell->vertex_dof_index(v, d));
                           // total_disp_vector(cell->vertex_dof_index(v, d)) += vertex_displacement[d];
                          }
                        
                      }
                    }
              }
            max_stress(cell->active_cell_index())=VM_stress_max;
            norm_of_stress(cell->active_cell_index()) =
              (accumulated_stress / quadrature_formula.size()).norm();
            pressure_bearings(cell->active_cell_index()) =
              (accumulated_pressure/  quadrature_formula.size()).norm();
              von_mises_stress(cell->active_cell_index()) =
              (local_VM_stress_values_at_qpoints); ///  quadrature_formula.size()).norm();
            
            //for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
             //   {
             //       count_faces +=1;
              //      face_index[cell->active_cell_index()](count_faces_internal)=count_faces;
                    
                // Iterate over all faces of the cell

              //  }
          }
        else
          norm_of_stress(cell->active_cell_index()) = -1e+20;
        


    //std::cout << "Faces total=" << count_faces<< std::endl;
    data_out.add_data_vector(norm_of_stress, "norm_of_stress");
    data_out.add_data_vector(pressure_bearings, "pressure_bearings");
    data_out.add_data_vector(centerx, "centerx");
    data_out.add_data_vector(centery, "centery");
    data_out.add_data_vector(centerz, "centerz");
    data_out.add_data_vector(total_displacement, "total_displacement");
    data_out.add_data_vector (von_mises_stress, "Von_Mises_stress");
    data_out.add_data_vector (max_stress, "max_stress");
    if (prm.eigenmodes)
    {
        for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
            data_out.add_data_vector(eigenfunctions[i],
                               std::string("eigenfunction_") +
                                 Utilities::int_to_string(i));
    }
    std::vector<types::subdomain_id> partition_int(
      triangulation.n_active_cells());
    GridTools::get_subdomain_association(triangulation, partition_int);
    //std::vector<types::boundary_id> boundary_vertices(triangulation.n_active_cells());

    const Vector<double> partitioning(partition_int.begin(),
                                      partition_int.end());

        //const Vector<double> boundary_nodes(boundary_nodes.begin(),
        //                              boundary_nodes.end());                                  
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();
   

    const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(
      "./"+prm.outputfolder+"/", "solution", timestep_no, mpi_communicator, 12);

    //const std::string pvtu_face_filename = data_out_face.write_vtu_with_pvtu_record(
     // "./", "boundary_id",  mpi_communicator, 12);

    

    const std::string filename_h5 = prm.outputfolder+"/solution_" + std::to_string(present_time) + ".h5";
    DataOutBase::DataOutFilterFlags flags(true, true);
    DataOutBase::DataOutFilter data_filter(flags);
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter, filename_h5, MPI_COMM_WORLD);
    //data_out_face.write_hdf5_parallel(data_filter, filename_h5, MPI_COMM_WORLD);
    HDF5::File data_file(filename_h5, HDF5::File::FileAccessMode::open, MPI_COMM_WORLD);
    //data_file.write_dataset("faces_index",face_index);
   // const std::string filename_h5_face = "solution_boundary_ids_" + std::to_string(present_time) + ".h5";


   // DataOutBase::DataOutFilterFlags flags(true, true);
    //DataOutBase::DataOutFilter data_filter(flags);
    //DataOutFaces::DataOutFilterFlags flags(true, true);
    //DataOutFaces::DataOutFilter data_filter(flags);
   // data_out_face.write_filtered_data(data_filter);
    //data_out_face.write_hdf5_parallel(data_filter, filename_h5_face, MPI_COMM_WORLD);
   // HDF5::File data_file_face(filename_h5_face, HDF5::File::FileAccessMode::open, MPI_COMM_WORLD);

    if (this_mpi_process == 0)
      {
        static std::vector<std::pair<double, std::string>> times_and_names;
        times_and_names.push_back(
          std::pair<double, std::string>(present_time, pvtu_filename));
        std::ofstream pvd_output(prm.outputfolder+"/solution.pvd");
        DataOutBase::write_pvd_record(pvd_output, times_and_names);


      //if (present_time == prm.delta_t)
       // {
          //std::ofstream out("boundary_ids.vtu");
          //data_out_face.write_vtu(out);
      //  }

      }
  }
 
 
 
 
  template <int dim>
  void TopLevel<dim>::do_initial_timestep()
  {
    //TimerOutput::Scope t(computing_timer, "do_initial_timestep");
    
    present_time += present_timestep;
    ++timestep_no;
    pcout << "Timestep " << timestep_no << " at time " << present_time
          << std::endl;
 
    for (unsigned int cycle = 0; cycle < prm.n_refinement_cycles; ++cycle)
      {
        pcout << "  Cycle " << cycle << ':' << std::endl;

        if (cycle == 0  )
          create_coarse_grid();
        else 
          refine_initial_grid();
 
        pcout << "    Number of active cells:       "
              << triangulation.n_active_cells() << " (by partition:";
        for (unsigned int p = 0; p < n_mpi_processes; ++p)
          pcout << (p == 0 ? ' ' : '+')
                << (GridTools::count_cells_with_subdomain_association(
                     triangulation, p));
        pcout << ')' << std::endl;
 
        setup_system();
 
        pcout << "    Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (by partition:";
        for (unsigned int p = 0; p < n_mpi_processes; ++p)
          pcout << (p == 0 ? ' ' : '+')
                << (DoFTools::count_dofs_with_subdomain_association(dof_handler,
                                                                    p));
        pcout << ')' << std::endl;
        if (prm.eigenmodes && cycle == 0)
            //solve_modes_serial();
            solve_modes(prm.eigensolver, prm.eigenprecond);

        if (prm.useNR) 
          {
            solve_timestep_newton();
          }
        else
          {
            solve_timestep();
          }

        
        

      }
  }
 
 
 
 
  template <int dim>
  void TopLevel<dim>::do_timestep()
  {
    //TimerOutput::Scope t(computing_timer, "do_timestep");
    
      present_time += present_timestep;
      ++timestep_no;
      pcout << "Timestep " << timestep_no << " at time " << present_time
          << std::endl;
    

    if (present_time > end_time)
      {
        present_timestep = -(present_time - end_time);
        present_time = end_time;
        

      }
 
    //setup_system();
     if (prm.useNR)
      {
        solve_timestep_newton();
      }
    else
      {
        solve_timestep();
      }
 
    if (!prm.useNR && prm.dynamicmode)
      {
       move_mesh(displacement_locally_relevant_solution);
      }
    else if(!prm.useNR && prm.dynamicmode)
      {
        move_mesh(locally_relevant_solution);
      }

    output_results();
    if (present_time >= end_time)
      if (this_mpi_process==0)
          {
            std::string command = "python3 " + prm.postproscriptpath;
            system(command.c_str());
          }
    
    pcout << std::endl;
  }
 
 
 
  template <int dim>
  void TopLevel<dim>::refine_initial_grid()
  {
      
      //Vector<double> temp_disp;
      //temp_disp.reinit(dof_handler.n_dofs());
      //temp_disp = locally_relevant_solution;
     TimerOutput::Scope t(computing_timer, "refine_initial_grid");
    Vector<float> error_per_cell(triangulation.n_active_cells());
    //if (prm.dynamicmode)
    //  KellyErrorEstimator<dim>::estimate(
    //    dof_handler,
    //    QGauss<dim - 1>(fe.degree + 1),
    //    std::map<types::boundary_id, const Function<dim> *>(),
    //    displacement_n,
    //    error_per_cell,
    //    ComponentMask(),
    //    nullptr,
    //    MultithreadInfo::n_threads(),
    //    this_mpi_process);
    //else
      KellyErrorEstimator<dim>::estimate(
        dof_handler,
        QGauss<dim - 1>(fe.degree + 1),
        std::map<types::boundary_id, const Function<dim> *>(),
        locally_relevant_solution,
        error_per_cell,
        ComponentMask(),
        nullptr,
        MultithreadInfo::n_threads(),
        this_mpi_process);
 
    const unsigned int n_local_cells =
      triangulation.n_locally_owned_active_cells();
 
    PETScWrappers::MPI::Vector distributed_error_per_cell(
      mpi_communicator, triangulation.n_active_cells(), n_local_cells);
 
    for (unsigned int i = 0; i < error_per_cell.size(); ++i)
      if (error_per_cell(i) != 0)
        distributed_error_per_cell(i) = error_per_cell(i);
    distributed_error_per_cell.compress(VectorOperation::insert);
 
    error_per_cell = distributed_error_per_cell;
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, error_per_cell, 0.1, 0.03);
    triangulation.execute_coarsening_and_refinement();
 
    setup_quadrature_point_history();
  }
 
 
 
 
  template <int dim>
   void TopLevel<dim>::move_mesh (const PETScWrappers::MPI::Vector &displacement) const
  {
    //TimerOutput::Scope t(computing_timer, "move_mesh");
    pcout << "    Moving mesh..." << std::endl;
    Vector<double> temp_disp;
      temp_disp.reinit(dof_handler.n_dofs());
      temp_disp = displacement;
     std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
      for (auto &cell : dof_handler.active_cell_iterators())
        for (const auto v : cell->vertex_indices())
          if (vertex_touched[cell->vertex_index(v)] == false)
            {
              vertex_touched[cell->vertex_index(v)] = true;

              Point<dim> vertex_displacement;
              for (unsigned int d = 0; d < dim; ++d)
                vertex_displacement[d] = displacement(cell->vertex_dof_index(v, d));

              cell->vertex(v) += vertex_displacement;
            }
      
    }
 
 
 
  template <int dim>
  void TopLevel<dim>::setup_quadrature_point_history()
  {
    TimerOutput::Scope t(computing_timer, "setup_quadrature_point_history");
   //triangulation.clear_user_data();
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    unsigned int our_cells = 0;
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      if (cell->subdomain_id() == this_mpi_process)
        ++our_cells;

    {
      std::vector<PointHistory<dim>> tmp;
      quadrature_point_history.swap(tmp);
    }
   // quadrature_point_history.resize(our_cells * quadrature_formula.size());
    quadrature_point_history.resize(
      triangulation.n_locally_owned_active_cells() * quadrature_formula.size());
    unsigned int history_index = 0;
    for (auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
    //for (; cell != endc; ++cell)
    //  if (cell->is_locally_owned())
        {
          cell->set_user_pointer(&quadrature_point_history[history_index]);
          history_index += quadrature_formula.size();
        }
 
    Assert(history_index == quadrature_point_history.size(),
           ExcInternalError());
  }
 
 
 
 
 
  template <int dim>
  void TopLevel<dim>::update_quadrature_point_history()
  {
    TimerOutput::Scope t(computing_timer, "update_quadrature_point_history");
    auto stress_strain_tensor_ = get_stress_strain_tensor<dim>(prm.lambda, prm.mu);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients);
 
    std::vector<std::vector<Tensor<1, dim>>> displacement_increment_grads(
      quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));
 
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    //const FEValuesExtractors::Vector displacement(0);
    
    for (;  cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          PointHistory<dim> *local_quadrature_points_history =
            reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
          Assert(local_quadrature_points_history >=
                   &quadrature_point_history.front(),
                 ExcInternalError());
          Assert(local_quadrature_points_history <
                   &quadrature_point_history.back(),
                 ExcInternalError());
 
          fe_values.reinit(cell);
          if (prm.dynamicmode)
            {
                
                fe_values.get_function_gradients(displacement_locally_relevant_solution,
                                           displacement_increment_grads);      
               // fe_values.
            }  
          else  
            fe_values.get_function_gradients(locally_relevant_solution,
                                           displacement_increment_grads);
           
          for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
            {

                //local_quadrature_points_history[q].displacement = locally_relevant_solution(q);
              const SymmetricTensor<2, dim> new_stress =
                (local_quadrature_points_history[q].old_stress +
                 (stress_strain_tensor_ *
                  get_strain(displacement_increment_grads[q])));
 
              const Tensor<2, dim> rotation =
                get_rotation_matrix(displacement_increment_grads[q]);
 
              const SymmetricTensor<2, dim> rotated_new_stress =
                symmetrize(transpose(rotation) *
                           static_cast<Tensor<2, dim>>(new_stress) * rotation);
 
              local_quadrature_points_history[q].old_stress =
                rotated_new_stress;
            }
        }
  }
 
} // namespace Step18
 
 
int main(int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace ParametricShaft;
 
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      if (argc < 2) 
      {
        
        std::cout << "Using default parameters.xml parameter file"<<std::endl;
        TopLevel<3> elastic_problem("parameters.xml");
        elastic_problem.run(); 

      }
      else{
        std::string param_file = argv[1];
        std::cout << "Using "<< param_file << " parameter file"<<std::endl;
        TopLevel<3> elastic_problem(param_file);
        elastic_problem.run();        
      }
      
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
 
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  
  return 0;
}