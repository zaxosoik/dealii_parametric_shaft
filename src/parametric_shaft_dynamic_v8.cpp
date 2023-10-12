
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
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
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
 
#include <cstdlib> 

//namespace LA
//{
//#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
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
const double rpm_to_rps = pi/30; // Conversion factor from RPM to RPS
namespace ParametricShaft
{
  
 
  namespace Parameters
  {

    struct Geometry
    {
      double radius;
      double half_length;
      // double bearing_x;
      // double bearing_length;
      unsigned int partitions;
      unsigned int n_global_refinements;

      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);

    };
    void Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Radius",
                          "0.5",
                          Patterns::Double(0.0),
                          "Radius of the cylinder");  
        prm.declare_entry("Half length",
                          "10.0",
                          Patterns::Double(0.0),
                          "Half length of the cylinder");
 
                      
        prm.declare_entry("Partitions",
                          "10",
                          Patterns::Integer(1),
                          "Number of partitions");
        prm.declare_entry("Global refinements",
                          "2",
                          Patterns::Integer(1),
                          "Number of global refinements");
      }
      prm.leave_subsection();
    }
    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        radius = prm.get_double("Radius");
        half_length = prm.get_double("Half length");
        // bearing_x = prm.get_double("Bearing x");
        // bearing_length = prm.get_double("Length of the Bearing");
        partitions = prm.get_integer("Partitions");
        n_global_refinements = prm.get_integer("Global refinements");

      }
      prm.leave_subsection();
    }

    struct Engine
    { 
      double MCR;
      double rpm_MCR;
      double rpm;
      double v_ship;

      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
    void Engine::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Engine");
      {
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
        prm.declare_entry("VSHIP",
                          "20.0",
                          Patterns::Double(0.0),
                          "Speed of the ship (knots)");

      }
      prm.leave_subsection();
    }
    void Engine::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Engine");
      {
        MCR = 1000*prm.get_double("MCR");
        rpm_MCR = prm.get_double("RPM MCR");
        rpm = prm.get_double("RPM");
        v_ship = prm.get_double("VSHIP")*0.5144444;
      }
      prm.leave_subsection();
    }

    struct Bearings
    {
      unsigned int bearings;
      double bearing_x;
      double bearing_length;
      double bearing_diameter;
      double eccentricity;
      double bearing_phi;
      double bearing_tol;
      double viscocity;

      double displacement;
      unsigned int direction;

      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
    };
    void Bearings::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Bearings");
      {
        prm.declare_entry("bearings",
                          "1",
                          Patterns::Integer(1),
                          "Number of bearings");
        prm.declare_entry("Bearing x",
                          "0.5",
                          Patterns::Double(0.0),
                          "x coordinate of the bearing");
        prm.declare_entry("Length of the Bearing",
                          "1.0",
                          Patterns::Double(0.0),
                          "Length of the bearing");
        prm.declare_entry("Diameter Bearing",
                          "1.05",
                          Patterns::Double(0.0),
                          "Diameter of the bearing");
        prm.declare_entry ("Bearing Eccentricity",
                          "0.0001",
                          Patterns::Double(0.0),
                          "Eccentricity of Bearing");
        prm.declare_entry ("Bearing phi",
                          "3.66519",
                          Patterns::Double(0.0),
                          "Angle of max eccentricity in rad");
        prm.declare_entry("Viscocity",
                          "0.001",
                          Patterns::Double(0.0),
                          "Viscocity of the bearing");
        prm.declare_entry("Bearing tol",
                          "0.0001",
                          Patterns::Double(0.0),
                          "Tolerance for solving the bearing");
        prm.declare_entry("Bearing Displacement",
                          "0.01",
                          Patterns::Double(0.0),
                          "Displacement of the bearing");
        prm.declare_entry("Direction",
                          "2",
                          Patterns::Selection("1|2|3"),
                          "Direction of the bearing Displacement");


      }
      prm.leave_subsection();
    }
    void Bearings::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Bearings");
      {
        bearings = prm.get_integer("bearings");
        bearing_x = prm.get_double("Bearing x");
        bearing_length = prm.get_double("Length of the Bearing");
        bearing_diameter = prm.get_double("Diameter Bearing");
        viscocity = prm.get_double("Viscocity");
        displacement = prm.get_double("Bearing Displacement");
        direction = prm.get_integer("Direction")-1;
        eccentricity = prm.get_double("Bearing Eccentricity");
        bearing_phi = prm.get_double("Bearing phi");
        bearing_tol = prm.get_double("Bearing tol");
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
                          Patterns::Selection("KrylovSchur|GeneralizedDavidson|JacobiDavidson|Lanczos"),
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
                           public Engine,
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
      Engine::declare_parameters(prm);
      Bearings::declare_parameters(prm);
      Solver::declare_parameters(prm);
      Materials::declare_parameters(prm);
      Time::declare_parameters(prm);

    }
 
    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      Geometry::parse_parameters(prm);
      Engine::parse_parameters(prm);
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
  };
 
 
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
 
    void setup_system();
    void compute_dirichlet_constraints();
    
    // FOR ALL TIMESTEPS
    void assemble_system();
    void assemble_bearing_rhs(const unsigned int bearing_boundary_id, const double eccentricity, const double phi);
    void predictors();
    void time_stepping();
 
    void solve_timestep();
 
    unsigned int solve_linear_problem();
    void assemble_res_newton();
    void newton();
    void solve_modes(std::string solver_name, std::string preconditioner_name);
 
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
    PETScWrappers::MPI::Vector newton_res;
    PETScWrappers::MPI::Vector newton_res_prev;

    std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
    std::vector<double>                eigenvalues;
    
    PETScWrappers::MPI::Vector bearing_rhs;
    PETScWrappers::MPI::Vector bearing_locally_relevant_solution;
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
    values(direction) = present_timestep * velocity;
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
                              Vector<double> &  values_displacement) const ;
    

 
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
    , velocity(displacement/(end_time))
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
        double power = C * std::pow(rps, 3); // Calculate power using the propeller law
        double torque = power / (2 * pi * rps); // Calculate torque
        return torque;
    }
    // Function to convert RPM to Force
    double rpmToForce(double rpm, double C, double velocity) {
        double rps = rpm * rpm_to_rps; // Convert RPM to RPS
        double power = C * std::pow(rps, 3); // Calculate power using the propeller law
        double force = power / velocity; // Calculate force
        return force;
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
    values(0) = pressure;

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
      PropellerTorque(const double torque, const double polar_moment);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override;
      virtual void 
      vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;
      
    private:
      const double torque; 
      const double polar_moment;
  };

  template <int dim>
  PropellerTorque<dim>::PropellerTorque(const double torque, const double polar_moment)
    : Function<dim>(dim), torque(torque), polar_moment(polar_moment)
  {}

  template <int dim>
  inline void PropellerTorque<dim>::vector_value(const Point<dim> &p,
                                                Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);
    double y = p(1);
    double z = p(2);
    double distance = sqrt(y*y+z*z);
    double angle = atan2(z,y);
    double tau = -torque*distance/(polar_moment);
    values = 0;
    values(1) = tau* sin(angle);
    values(2) = tau* cos(angle);

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
    values(2) = omega*distance*cos(angle);

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
//AssertDimension(values.size(), dim);
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
   
  values_h = sqrt(ins_sqrt)-bearing_radius; //h
}
template <int dim>
void BearingSurfaceMove<dim>::h_dot_vector_value(const Point<dim> &p,
                                            Vector<double> &values_h_dot) const
{
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
    double ins_sqrt_dot = -2*y_dot-2*z_dot;
    values_h_dot = pow((ins_sqrt),(-1/2))*(ins_sqrt_dot); //h_dot

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
//template <int dim>
//class SolveBearing : public Function<dim>
//  {
//    public:
//      SolveBearing(Triangulation<dim> &triangulation, 
//                    DoFHandler<dim> &dof_handler,
//                    const double eccentricity, 
//                    const double mu, 
//                    const double bearing_radius,
//                    //const double bearing_length,
//                    const double rpm,
//                    const unsigned int boundary_id);
//      virtual void vector_value(const Point<dim> &p,
//                                Vector<double> &values) const override;
//    private:
//        Triangulation<dim> &triangulation;
//        DoFHandler<dim> &dof_handler;
//        const double eccentricity;
//        const double mu;
//        const double bearing_radius;
//        //const double bearing_length;
//        const double rpm;
//        const unsigned int boundary_id;
//  };
//  template <int dim>
//  SolveBearing<dim>::SolveBearing(Triangulation<dim> &triangulation,
//                                  DoFHandler<dim> &dof_handler, 
//                                  const double eccentricity, 
//                                  const double mu, 
//                                  const double bearing_radius,
//                                  //const double bearing_length,
//                                  const double rpm,
//                                  const unsigned int boundary_id)
//                              :Function<dim>(dim), 
//                                triangulation(triangulation),
//                                dof_handler(dof_handler),
//                                eccentricity(eccentricity),
//                                mu(mu),
//                                bearing_radius(bearing_radius),
//                                //bearing_length(bearing_length),
//                                rpm(rpm),
//                                boundary_id(boundary_id)
//    {
//
//      
//    }
//    template <int dim>
//    void SolveBearing<dim>::vector_value(const Point<dim> &p,
//                                          Vector<double> &values) const
//    {
//      
//
//      IndexSet boundary_dof_set;
//      std::vector<types::global_dof_index> face_dof_indices(fe.dofs_per_face);
//      //DoFTools::extract_boundary_dofs(dof_handler,
//      //                              ComponentMask(), 
//      //                              
//      //                              boundary_id);
//     for (const auto &cell : dof_handler.active_cell_iterators())
//      {
//      if (cell->is_locally_owned())
//        for (const auto &face : cell->face_iterators())
//        {
//          if (face->boundary_id() == boundary_id)
//          {
//            face->get_dof_indices(face_dof_indices);
//            for (auto dof_index : face_dof_indices)
//            {
//              boundary_dof_set.add_index(dof_index);
//            }
//          }
//        }
//      }
//    }
//
//
  template <int dim>
  TopLevel<dim>::TopLevel(const std::string &input_file)
    : prm(input_file)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , triangulation(MPI_COMM_WORLD/*,
                    typename Triangulation<dim>::MeshSmoothing(
                        Triangulation<dim>::smoothing_on_refinement |
                        Triangulation<dim>::smoothing_on_coarsening)*/)
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

    const double radius = prm.radius;
    const double half_length = prm.half_length;
    const double bearing_x = prm.bearing_x;
    const double bearing_length = prm.bearing_length;
    GridGenerator::subdivided_cylinder(triangulation, prm.partitions, radius, half_length);

  
    GridTools::partition_triangulation (n_mpi_processes,
                                           triangulation);
      
      const double x0 = -prm.half_length;
      const double x1 =  prm.half_length;
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
    triangulation.refine_global(prm.n_global_refinements);  
    {
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
 
            if (dist_vector[0] == half_length)
               cell->face(f)->set_boundary_id(0);
            else if (dist_vector[0] == -half_length)
               cell->face(f)->set_boundary_id(1);
     
            else if (std::sqrt(dist_vector[1] * dist_vector[1] +
                              dist_vector[2] * dist_vector[2]) <
                    radius)
                    {
                    if (dist_vector[0] <(bearing_x+bearing_length/2) && dist_vector[0]>(bearing_x-bearing_length/2))
                       cell->face(f)->set_boundary_id(3);  
                    else    
                       cell->face(f)->set_boundary_id(2);
                    }
                                  
            else
               cell->face(f)->set_boundary_id(4);
          }  
        }   
    
    }

    setup_quadrature_point_history();
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

    newton_res.reinit(locally_owned_dofs, mpi_communicator);
    newton_res_prev.reinit(locally_owned_dofs, mpi_communicator);


    

    locally_relevant_solution.reinit(locally_owned_dofs,
                                       locally_relevant_dofs,
                                       mpi_communicator);

    bearing_locally_relevant_solution.reinit(locally_owned_dofs,
                                       locally_relevant_dofs,
                                       mpi_communicator);

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
        //component_mask[1] = false;
        //component_mask[2] = false;
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                     0,
        //                                     Functions::ZeroFunction<dim>(dim),
        //                                     constraints_dirichlet_and_hanging_nodes,
        //                                     component_mask); 
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                     0,
        //                                     AccellerationAngularBoundary<dim>(rpm_to_rps*prm.rpm),
        //                                    constraints_dirichlet_and_hanging_nodes
        //                                    ,component_mask);      
       // component_mask[0] = true;
       // component_mask[1] = true;
       // component_mask[2] = true;
       // VectorTools::interpolate_boundary_values(dof_handler,
       //                                         0,
       //                                         AccellerationAngularBoundary<dim>(prm.rpm*rpm_to_rps),
       //                                         constraints_dirichlet_and_hanging_nodes,
       //                                         component_mask);              
    }
    else
    {
        //component_mask[0] = false;
        //component_mask[1] = false;
        //component_mask[2] = true;
        //VectorTools::interpolate_boundary_values(dof_handler,
        //                                        3,
        //                                        IncrementalBoundaryValues<dim>(prm.end_time, present_timestep,prm.direction,prm.displacement),
        //                                        constraints_dirichlet_and_hanging_nodes,
        //                                        component_mask); 
        component_mask[0] = true;
        component_mask[1] = true;
        component_mask[2] = true;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                0,
                                                Functions::ZeroFunction<dim>(dim),
                                                constraints_dirichlet_and_hanging_nodes,
                                                component_mask);
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
    if (timestep_no==1)
    {
      displacement_n = 0;
      velocity_n = 0;
      acceleration_n = 0;
    }

      
    const double rpm = prm.rpm;
    
   
   
    //BearingForce<dim> bearing_force(rpm, prm.bearing_diameter, prm.viscocity);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    
  

    FEFaceValues<dim> fe_face_values(fe_values.get_fe(), 
                                            face_quadrature_formula,
                                           update_values | update_quadrature_points | update_JxW_values);
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
    
    double total_area = 0.0;
    Point<dim> weighted_position;
   // Vector<double> boundary_1();

    typename dealii::DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    unsigned int count_n_faces=0;
    //std::cout << "\n Assembly constants and matrices INITIALISED"<<std::endl;
    //for (const auto &cell : dof_handler.active_cell_iterators())
    
    for (; cell != endc; ++cell)
      //if (cell->subdomain_id() == 0)
        {
          
            //if (cell->at_boundary())
            {
                for (unsigned int face = 0; face < n_faces_per_cell; ++face)
                {
                    if (cell->face(face)->boundary_id() == 1)
                    {
                        count_n_faces+=1;
                        fe_face_values.reinit(cell, face);
                        fe_values.reinit(cell);
                        for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                        {
                            double JxW = fe_face_values.JxW(q);
                            total_area += JxW;
                            weighted_position += JxW* fe_face_values.quadrature_point(q);
                        }
                  }
              }
          }
        }


     std::vector<bool> faces_touched(count_n_faces, false);

    const double total_force = rpmToForce(rpm, prm.MCR/pow(prm.rpm_MCR*rpm_to_rps,3), prm.v_ship);
    const double total_torque = rpmToTorque(rpm,prm.MCR/pow(prm.rpm_MCR*rpm_to_rps,3));
    Point<dim> centroid_;
    const double pressure = total_force / total_area;
    Vector<double> distance2centroid(n_q_points);
    Point<dim> centroid2rest;

    Vector<double> angle4force(n_q_points);

    if(total_area > 0.0)
    {
        centroid_ = weighted_position / total_area;
    }
    centroid2rest = centroid_;
    double polar_moment = 0.0;
    cell = dof_handler.begin_active();
    //for (const auto &cell : dof_handler.active_cell_iterators())
      //  if (cell->is_locally_owned())
    //if (this_mpi_process==0)
    for (; cell != endc; ++cell)
     //if (cell->subdomain_id() == 0)
        {
            //if (cell->at_boundary())
            {
                for (unsigned int face = 0; face < n_faces_per_cell; ++face)
                {
                    if (cell->face(face)->boundary_id() == 1)
                    {
                        fe_face_values.reinit(cell, face);
                        
                        for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
                        {
                            //propeller_pressure[q](0) = pressure;
                            Point<dim> q_point = fe_face_values.quadrature_point(q);
                            Point<dim> relative_position(q_point - centroid_);
                            distance2centroid[q]=  sqrt(relative_position(2)*relative_position(2)+relative_position(1)*relative_position(1));
                            Point<dim> pos = q_point;
                            //centroid2rest +=pos*fe_face_values.JxW(q)
                            angle4force[q] = atan2(relative_position(2),relative_position(1));
                            //distance2centroid[q]= relative_position(3);
                            polar_moment += fe_face_values.JxW(q) * (relative_position.square());
                            //std::cout << distance2centroid[q]<< std::endl;
                        }
                    }
                }
            }
        }
    
    if (polar_moment > 0)
      {
        polar_moment = polar_moment;
      }
      else
      {
        polar_moment = pi*pow(prm.radius,4)/32;
      }

    //if (this_mpi_process==0)
    pcout <<  "\n Total Area of section: "<< total_area << "\n Pressure=  " << pressure <<" for RPM="<<prm.rpm << "\n Centroid=  " << centroid_ << "\n Polar Moment of Inertia J=  " << polar_moment<<  std::flush;

    auto stress_strain_tensor_ = get_stress_strain_tensor<dim>(prm.lambda, prm.mu);
    /*
    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->at_boundary())
        {
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
            {
                if (cell->face(face)->boundary_id() == 1)
                {
                    fe_face_values.reinit(cell, face);
                    cell->get_dof_indices(local_dof_indices);
                    for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
                    {
                        boundary_pressure[local_dof_indices[i]] += pressure * fe_face_values.JxW(face)
                                                                *fe_face_values.shape_value(i, face);
                    }
                }
            }
        }*/
      pcout << "    Starting to  construct matrices for " << this_mpi_process <<" process done." <<std::endl;
    //for (const auto &cell : dof_handler.active_cell_iterators()) 
      //if (cell->is_locally_owned())
    unsigned int count=0;
    unsigned int countboundary1=0;
    unsigned int count_n_faces_2=0;
      BodyForce<dim>              body_force(prm.rho, prm.g);
    std::vector<Vector<double>> body_force_values(n_q_points,
                                                  Vector<double>(dim));
    VelocityAngular<dim> velocity_angular(prm.rpm*rpm_to_rps); 
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
    PropellerTorque<dim> propeller_torque(total_torque, polar_moment); 
    std::vector<Vector<double>> propeller_torque_values(n_q_points,
                                                  Vector<double>(dim));
    //SolveBearing<dim> bearing(triangulation,dof_handler, 
    //              /*const double eccentricity,      */0,
    //              /*const double mu, */ 0,
    //              /*const double bearing_radius,*/ 0.5,
    //              /*//const double bearing_length,*/ 
    //              /*const double rpm,*/ prm.rpm*rpm_to_rps,
    //              /*const unsigned int boundary_id*/ 3);

    const FEValuesExtractors::Vector displacement(0);
    cell = dof_handler.begin_active();
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
         
          
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                  const SymmetricTensor<2, dim>
                    eps_phi_i = get_strain(fe_values, i, q_point),
                    eps_phi_j = get_strain(fe_values, j, q_point);
 
                  cell_matrix(i, j) += (eps_phi_i *            
                                        stress_strain_tensor_ * 
                                        eps_phi_j              
                                        ) *                    
                                       fe_values.JxW(q_point); 

                  

                  
                  cell_mass_matrix(i, j) += (prm.rho *
                                           fe_values.shape_value(i, q_point) *
                                            fe_values.shape_value(j, q_point) *
                                            fe_values.JxW(q_point));
                  cell_damping_matrix(i, j) += (prm.mu_rayleigh*cell_mass_matrix(i, j) + prm.lambda_rayleigh*cell_matrix(i, j));
                  cell_stiffness_matrix(i, j) +=  cell_mass_matrix(i, j) +   prm.gamma*prm.delta_t*(cell_damping_matrix(i, j)) + prm.beta*prm.delta_t*prm.delta_t*cell_matrix(i, j);     
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

          DynamicIncrementalValues<dim> dynamicincrementalvalues_bearing1(prm.end_time, present_timestep,prm.direction,prm.displacement);
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
                      cell_res(i) += -old_stress * get_strain(fe_values, i, q_point);
                      if (prm.dynamicmode)
                        cell_rhs(i) +=
                            (body_force_values[q_point](component_i)*fe_values.shape_value(i, q_point) 
                            )
                            * fe_values.JxW(q_point);
                      else
                         cell_rhs(i) +=
                            (body_force_values[q_point](component_i)*fe_values.shape_value(i, q_point) 
                            +cell_res(i))
                            * fe_values.JxW(q_point);

                    if (timestep_no==1)
                     {
                        cell_velocity(i)     += fe_values.shape_value(i, q_point) *velocity_angular_values[q_point](component_i)    *fe_values.JxW(q_point);
                        cell_accel(i)        += fe_values.shape_value(i, q_point) *acceleration_angular_values[q_point](component_i)*fe_values.JxW(q_point);
                        
                        //if (timestep_no>1)
                        //cell_displacement(i) += fe_values.shape_value(i, q_point) *displacement_engine_values[q_point](component_i) *fe_values.JxW(q_point);
                    }

                }
            }

          
          
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
            {
              fe_face_values.reinit(cell, face);
              //fe_values.reinit(cell);
              //if (faces_touched[])
              if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() == 1))
              //if (cell->boundary_id() == 1)
                  {
                         

  
                          propeller_pressure.vector_value_list(fe_face_values.get_quadrature_points(), 
                                                      propeller_pressure_values);
                          
                          propeller_torque.vector_value_list(fe_face_values.get_quadrature_points(), 
                                                      propeller_torque_values);

                          

                            for (unsigned int q_point = 0; q_point < n_faces_per_cell;
                                ++q_point)
                              {
                                Tensor<1, dim> rhs_values;
                                for (unsigned int i = 0; i < dim; ++i)
                                  {
                                    rhs_values[i] = propeller_torque_values[q_point][i]+propeller_pressure_values[q_point][i];
                                  }
                                for (unsigned int i = 0; i < dofs_per_cell; ++i)            
                                  {
                                        cell_rhs(i) += (fe_values[displacement].value(i, q_point)) * rhs_values
                                              * fe_values.JxW(q_point);
                                                  countboundary1 +=1;


                                  }
                                }
                   }
                  else if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() == 0) && (prm.dynamicmode))
              //if (cell->boundary_id() == 1)
                  {
                         

  
                          propeller_pressure.vector_value_list(fe_face_values.get_quadrature_points(), 
                                                      propeller_pressure_values);
                          
                          propeller_torque.vector_value_list(fe_face_values.get_quadrature_points(), 
                                                      propeller_torque_values);

                          

                            for (unsigned int q_point = 0; q_point < n_faces_per_cell;
                                ++q_point)
                              {
                                Tensor<1, dim> rhs_values;
                                for (unsigned int i = 0; i < dim; ++i)
                                  {
                                    rhs_values[i] = -propeller_torque_values[q_point][i]-propeller_pressure_values[q_point][i];
                                  }
                                for (unsigned int i = 0; i < dofs_per_cell; ++i)            
                                  {
                                        cell_rhs(i) += (fe_face_values[displacement].value(i, q_point)) * rhs_values
                                              * fe_face_values.JxW(q_point);
                                                  countboundary1 +=1;


                                  }
                                }
                   }
                //else if (cell->face(face)->boundary_id() == 3 && prm.dynamicmode )
                //                {
//
                //                      dynamicincrementalvalues_bearing1.vector_value_list_displacement(fe_values.get_quadrature_points(), 
                //                                                  dynamicincremental_displacement_bearing1);
                //                      dynamicincrementalvalues_bearing1.vector_value_list_velocity(fe_values.get_quadrature_points(), 
                //                                                  dynamicincremental_velocity_bearing1);
//
                //                      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                //                            {
                //                              Tensor<1, dim> vel_rhs_values;
                //                              Tensor<1, dim> disp_rhs_values;
                //                              for (unsigned int i = 0; i < dim; ++i)
                //                                {
                //                                  vel_rhs_values[i] =  dynamicincremental_velocity_bearing1[q_point](i);
                //                                  disp_rhs_values[i] =  dynamicincremental_displacement_bearing1[q_point](i);
                //                                }
                //                              for (unsigned int i = 0; i < dofs_per_cell; ++i)            
                //                                {
                //                              
                //                                  cell_displacement(i) +=  fe_face_values[displacement].value(i, q_point)*disp_rhs_values*fe_face_values.JxW(q_point);
                //                                  cell_velocity(i) += fe_face_values[displacement].value(i, q_point)*vel_rhs_values*fe_face_values.JxW(q_point);
                //                                }
                //                             }
                //                 }
         //
                    //else if (cell->face(face)->boundary_id() != 0 && prm.dynamicmode)
                
              }
            //}
              
                        
                  
              
          //std::cout << "Cell Matrices Constructed for MPI Process"<< this_mpi_process << std::endl;  
        
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
                                                acceleration_n);
            constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_accel_rotation,
                                                local_dof_indices,
                                                acceleration_n);
            constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_res,
                                                local_dof_indices,
                                                newton_res);

                                    
          //}
//
        //std::cout << "Hanging Nodes Matrices Constructed for MPI Process" << this_mpi_process << std::endl;  
        }
    //if (count==countboundary1) 
     //   std::cout << "pressure=0 for "<< this_mpi_process << std::endl;
     //else
     //  std::cout << "pressure=0 for "<< count <<" in "<<countboundary1 <<std::endl;

    pcout << "    Matrices for  " << this_mpi_process <<" process done." <<std::endl;

    //system_matrix.compress(VectorOperation::add);
    //system_mass_matrix.compress(VectorOperation::add);
    //system_stiffness_matrix.compress(VectorOperation::add);
    //
    //system_damping_matrix.compress(VectorOperation::add);
    //TODO fix rayleigh damping matrix


 
    const FEValuesExtractors::Scalar          z_component(dim-1);
    const FEValuesExtractors::Scalar          x_component(0);
    //const FEValuesExtractors::Scalar          bearing_component(dim-1);

   
  
    /*VectorTools::interpolate_boundary_values(dof_handler,
                                             2,
                                             BearingBoundaryValues<dim>(rpm),
                                             boundary_values,
                                             fe.component_mask(bearing_component));  */  
     /*VectorTools::interpolate_boundary_values(dof_handler,
                                                 1, 
                                                 pressure_boundary_values, 
                                                 boundary_values,
                                                 fe.component_mask(x_component));*/

 
    PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);

    pcout << "    Boundaries initialised for  " << this_mpi_process <<" process done." <<std::endl;
    assemble_bearing_rhs(3,prm.eccentricity,prm.bearing_phi);
    pcout << "    Bearing Boundaries initialised." <<std::endl;
    PETScWrappers::MPI::Vector tmp_rhs(locally_owned_dofs, mpi_communicator);
    tmp_rhs = bearing_locally_relevant_solution;
    tmp_rhs.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    system_rhs.add(1,tmp_rhs);
    pcout << "    Bearing Boundaries added." <<std::endl; 


    if (prm.dynamicmode)
    {
        system_matrix.compress(VectorOperation::add);
        system_mass_matrix.compress(VectorOperation::add);
        system_damping_matrix.compress(VectorOperation::add);
        system_dynamic_matrix.compress(VectorOperation::add);

        displacement_n.compress(VectorOperation::add);
        velocity_n.compress(VectorOperation::add);
        acceleration_n.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
        
        pcout << "    Predictors for " << this_mpi_process <<" process done." <<std::endl;
        //MatrixTools::apply_boundary_values(
        //  boundary_values, system_matrix, tmp, displacement_n, false);
        //MatrixTools::apply_boundary_values(
        //  boundary_values, system_damping_matrix, tmp, velocity_n, false);  
        //    displacement_n*=(-1);
        //    velocity_n*=(-1);        
        //    system_matrix.vmult_add(system_rhs,displacement_n);
        //    system_damping_matrix.vmult_add(system_rhs,velocity_n);
        //    displacement_n*=(-1);
        //    velocity_n*=(-1);
        
        //system_matrix*=prm.beta*prm.delta_t*prm.delta_t;
        //system_matrix.add(1,system_mass_matrix);
        //system_matrix.add(prm.gamma*prm.delta_t,system_damping_matrix);
        system_matrix.compress(VectorOperation::add);
        system_mass_matrix.compress(VectorOperation::add);
        system_damping_matrix.compress(VectorOperation::add);
        system_dynamic_matrix.compress(VectorOperation::add);

        displacement_n.compress(VectorOperation::add);
        velocity_n.compress(VectorOperation::add);
        acceleration_n.compress(VectorOperation::add);
        acceleration_n_prev.compress(VectorOperation::add);  
        //if (timestep_no>1)
        //  acceleration_n.add(-1,acceleration_n_prev);
        

        system_rhs.compress(VectorOperation::add);
        //system_matrix.compress(VectorOperation::add);
        pcout << "    System Matrix for " << this_mpi_process <<" process done." <<std::endl;        
        //system_mass_matrix.add(prm.gamma*prm.delta_t,system_damping_matrix);
        //system_mass_matrix.add(prm.beta*prm.delta_t*prm.delta_t,system_matrix);

        //system_matrix = system_stiffness_matrix;
        //system_matrix.compress(VectorOperation::add);
         std::map<types::global_dof_index, double> boundary_values;
        std::map<types::global_dof_index, double> boundary_values_dynamic;
        VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             AccellerationAngularBoundary<dim>(rpm_to_rps*prm.rpm),
                                            boundary_values_dynamic);    
        VectorTools::interpolate_boundary_values(dof_handler,
                                              0,
                                              Functions::ZeroFunction<dim>(0),
                                              boundary_values,
                                              fe.component_mask(x_component));                                      
        MatrixTools::apply_boundary_values(
         boundary_values, system_dynamic_matrix, tmp, system_rhs, false);
        locally_relevant_solution = tmp;
        predictors();
        pcout << "    Boundaries applied for " << this_mpi_process <<" process done." <<std::endl;
    }
    else
    {
         std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                3,
                                                IncrementalBoundaryValues<dim>(prm.end_time, present_timestep,prm.direction,prm.displacement),
                                                boundary_values); 
        system_matrix.compress(VectorOperation::add);
        system_mass_matrix.compress(VectorOperation::add);
        system_damping_matrix.compress(VectorOperation::add);
        system_dynamic_matrix.compress(VectorOperation::add);

        displacement_n.compress(VectorOperation::add);
        velocity_n.compress(VectorOperation::add);
        acceleration_n.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);

        MatrixTools::apply_boundary_values(
          boundary_values, system_matrix, tmp, system_rhs, false);
        locally_relevant_solution = tmp;
    }


   
    //pcout << "    Assembiling System for " << this_mpi_process <<" process done." <<std::endl;
  }
  
  template <int dim>
  void TopLevel<dim>::assemble_bearing_rhs(const unsigned int bearing_boundary_id, 
                                          const double eccentricity, const double phi)
  {
    {
     TimerOutput::Scope t(computing_timer, "Assembling Bearing");
    //std::vector<types::global_dof_index> face_dof_indices(fe.dofs_per_face);
    ////////////////////////////////
    //const IndexSet boundary_dof_set = DoFTools::extract_boundary_dofs(dof_handler,
    //                                                  boundary_id);
    //std::vector<bool> component_mask(dim);
    //const ComponentMask component_mask();
    std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert(bearing_boundary_id);
    //const std::set< types::boundary_id > boundary_id;
    //component_mask[0] = true;
    //component_mask[1] = true;
    //component_mask[2] = true;
    // std::vector<dealii::IndexSet> 
    //for (unsigned int i = 0; i < n_mpi_processes; ++i)

    const IndexSet boundary_dof_set = DoFTools::extract_boundary_dofs(dof_handler,ComponentMask(),boundary_ids);
  
    ////////////////////////////////
    //std::vector<types::global_dof_index> face_dof_indices(fe.dofs_per_face);
    //for (const auto &cell : dof_handler.active_cell_iterators())
    //{
    //  for (const auto &face : cell->face_iterators())
    //  {
    //    if (face->boundary_id() == boundary_id)
    //    {
    //      face->get_dof_indices(face_dof_indices);
    //      for (auto i : face_dof_indices)
    //      {
    //        boundary_dof_set.add_index();
    //      }
    //    }
    //  }
    //}


    //PETScWrappers::MPI::Vector bearing_rhs;
    //PETScWrappers::MPI::Vector bearing_locally_relevant_solution;
    //PETScWrappers::MPI::SparseMatrix bearing_system_matrix;

    //DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
    //DoFTools::make_sparsity_pattern(dof_handler,
    //                                sparsity_pattern,
    //                                constraints_dirichlet_and_hanging_nodes,
    //                                /*keep constrained dofs*/ false);
    //std::vector<dealii::IndexSet> locally_owned_dofs_per_processor =
    //DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
    //std::vector<dealii::types::global_dof_index> n_locally_owned_dofs(n_mpi_processes);
//
    //for (unsigned int i = 0; i < n_mpi_processes; ++i)
    //  n_locally_owned_dofs[i] = locally_owned_dofs_per_processor[i].n_elements();
    //SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
    //                                            dof_handler.locally_owned_dofs(),
    //                                           mpi_communicator,
    //                                           locally_relevant_dofs);
    //bearing_system_matrix.reinit(locally_owned_dofs,
    //                      locally_owned_dofs,
    //                      sparsity_pattern,
    //                      mpi_communicator);
//
    //      
    //bearing_rhs.reinit(locally_owned_dofs, mpi_communicator);
    //bearing_locally_relevant_solution.reinit(locally_owned_dofs,
    //                                  locally_relevant_dofs,
    //                                  mpi_communicator);                             
    //
    /*bearing_rhs.reinit(boundary_dof_set,mpi_communicator);
    bearing_locally_relevant_solution.reinit(boundary_dof_set,mpi_communicator);

    AffineConstraints<double> bearing_hanging_node_constraints;
    bearing_hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            bearing_hanging_node_constraints);
    DynamicSparsityPattern bearing_sparsity_pattern(boundary_dof_set);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    bearing_sparsity_pattern,
                                    bearing_hanging_node_constraints,
                                    false);
    bearing_hanging_node_constraints.close(); */
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    
  

    FEFaceValues<dim> fe_face_values(fe_values.get_fe(), 
                                            face_quadrature_formula,
                                           update_values | update_quadrature_points | update_JxW_values);
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
    
    double disp_increm = prm.displacement/prm.end_time*present_timestep;
    BearingSurfaceMove<dim> h(prm.bearing_diameter/2, eccentricity, phi,disp_increm,prm.direction,prm.delta_t);  
    std::vector<Vector<double>> h_values(n_q_points,Vector<double>(1));
    std::vector<Vector<double>> h_dot_values(n_q_points,Vector<double>(1));
   
    VelocityAngular<dim> velocity_angular(prm.rpm*rpm_to_rps); 
    std::vector<Vector<double>> velocity_angular_values(n_q_points,
                                                                  Vector<double>(dim));
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
              h.h_vector_value_list(fe_values.get_quadrature_points(), 
                                                  h_values);
              h.h_dot_vector_value_list(fe_values.get_quadrature_points(),
                                                  h_dot_values);


              velocity_angular.vector_value_list(fe_values.get_quadrature_points(), 
                                  velocity_angular_values);
              if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() == 1))
                {

                  
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      const unsigned int component_i =
                            fe.system_to_component_index(i).first;
                      const SymmetricTensor<2, dim>
                            eps_phi_i = get_strain(fe_values, i, q_point);
                      Tensor<1, dim> rhs_values; // = (eps_phi_i)*velocity_angular_values[q_point];
                      
                      for (unsigned int i = 0; i < dim; ++i)
                      {
                        rhs_values[i] = eps_phi_i[q_point][i]*velocity_angular_values[q_point][i];
                      }
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                          const SymmetricTensor<2, dim>
                            //eps_phi_i = get_strain(fe_values, i, q_point),
                            eps_phi_j = get_strain(fe_values, j, q_point);
                          cell_matrix(i, j) += (pow(h_values[q_point](0),3)/(12*prm.viscocity*1e-6))*(transpose(eps_phi_i) *            
                                                eps_phi_j              
                                                ) *                    
                                              fe_values.JxW(q_point);
                      }
                      
                      //cell_rhs(i) += (fe_face_values[displacement].value(i, q_point)) *h_values[q_point](0)*rhs_values[i] * fe_face_values.JxW(q_point)/2;
                      //cell_rhs(i) -= (fe_face_values[displacement].value(i, q_point)) *h_dot_values[q_point](0)* fe_face_values.JxW(q_point);
                      cell_rhs(i) += (fe_values.shape_value(i, q_point) ) *h_values[q_point](0)*rhs_values[i] * fe_values.JxW(q_point)/2;
                      cell_rhs(i) -= (fe_values.shape_value(i, q_point) ) *h_dot_values[q_point](0)* fe_values.JxW(q_point);
                    }
                }
                
                //break;                
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
    }
    pcout << "Cell Matrices Constructed for bearing with boundary_id:"<< bearing_boundary_id << std::endl;
    
    {
      TimerOutput::Scope t(computing_timer, "Solving Bearing");
      PETScWrappers::MPI::Vector distributed_locally_relevant_solution(
                    locally_owned_dofs, mpi_communicator);
      //distributed_locally_relevant_solution = 0;
      PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
      //data.symmetric_operator = true;

      SolverControl solver_control(dof_handler.n_dofs(), prm.bearing_tol* bearing_rhs.l2_norm()*1000);

      PETScWrappers::SolverCG cg(solver_control, mpi_communicator);
      PETScWrappers::PreconditionBoomerAMG preconditioner;
      preconditioner.initialize(bearing_system_matrix, data);
      cg.solve(bearing_system_matrix, distributed_locally_relevant_solution, bearing_rhs, preconditioner);
      pcout << "Bearing System Solved for bearing:"<< bearing_boundary_id <<" in "<<solver_control.last_step()<<" iterations"<< std::endl;
      constraints_dirichlet_and_hanging_nodes.distribute(distributed_locally_relevant_solution);
      bearing_locally_relevant_solution = distributed_locally_relevant_solution;
    }
    
       

  }

  template <int dim>
  void TopLevel<dim>::predictors()
  { 
    TimerOutput::Scope t(computing_timer, "predictors");
    // Update velocity and acceleration with Newmark-beta method PREDICTORS

    pcout << " Predictors calculations " << std::endl;    
        
    displacement_n.add(prm.delta_t,velocity_n);
    displacement_n.add((1/2-prm.beta)*prm.delta_t*prm.delta_t,acceleration_n);
    velocity_n.add((1-prm.gamma)*prm.delta_t,acceleration_n);

        //MatrixTools::apply_boundary_values(
        // boundary_values, tmp_matrix, tmp, displacement_n, false);

    //constraints_dirichlet_and_hanging_nodes.distribute(displacement_n);
    //constraints_dirichlet_and_hanging_nodes.distribute(velocity_n);
    //constraints_dirichlet_and_hanging_nodes.distribute(acceleration_n);
  }
  template <int dim>
  void TopLevel<dim>::time_stepping()
  {
    TimerOutput::Scope t(computing_timer, "time_stepping");
   // acceleration_n = distrivute
   //parallel::distributed::
   //SolutionTransfer<dim,VectorType,DoFHandler<dim,spacedim>>
   // sol_trans(dof_handler);
   // sol_trans.deserialize(distributed_vector);
    //acceleration_n =locally_relevant_solution;

    pcout << " Timestepping calculations " << std::endl;       
        

   //acceleration_n.distribute_loacl_to_global(locally_relevant_solution);

    // Update velocity and acceleration with Newmark-beta method CORRECTORS for the next time-step
    velocity_n.add((prm.gamma)*prm.delta_t,acceleration_n);
    displacement_n.add(prm.beta*prm.delta_t*prm.delta_t,acceleration_n);

//
    //locally_relevant_solution = displacement_n;
    
    //constraints_dirichlet_and_hanging_nodes.distribute(displacement_n);
    //constraints_dirichlet_and_hanging_nodes.distribute(velocity_n);
    //constraints_dirichlet_and_hanging_nodes.distribute(acceleration_n);
        

  }


  template <int dim>
  void TopLevel<dim>::assemble_res_newton()
  {
    ///////
    

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

    const unsigned int max_newton_iter = 100;

    for (unsigned int newton_step = 1; newton_step <= max_newton_iter; ++newton_step)
      {
        if (newton_step == 1 && ( timestep_no == 1))
            {
              acceleration_n_prev = 0;
              velocity_n_prev = 0;
              displacement_n_prev = 0;
              newton_res_prev = 0;
            }
        else
            {
              acceleration_n_prev = acceleration_n;
              velocity_n_prev = velocity_n;
              displacement_n_prev = displacement_n;
            }
        std::cout << "Newton Iteration: " << newton_step << std::endl;
        //{
        //  netwton_res = 
        //  newtown_rhs = systen
//
        //}
        //    
        }

  }

  template <int dim>
  void TopLevel<dim>::solve_timestep()
  {
    //TimerOutput::Scope t(computing_timer, "solve_timestep");

    pcout << "    Assembling system..." << std::flush;
    assemble_system();
    pcout << " norm of rhs is " << system_rhs.l2_norm() << std::endl;
    
    const unsigned int n_iterations = solve_linear_problem();
 
    pcout << "    Solver converged in " << n_iterations << " iterations."
          << std::endl;
 
    pcout << "    Updating quadrature point data..." << std::flush;
    update_quadrature_point_history();
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

          
    SolverControl solver_control(prm.max_iter*1000,
                                 prm.tol * system_rhs.l2_norm()*1000,
                                 /*log_history*/ true,
                                true);
 
    //PETScWrappers::SolverCG cg(solver_control, mpi_communicator);
//#ifdef USE_PETSC_LA
//    LA::SolverCG solver(solver_control, mpi_communicator);
//#else
//    LA::SolverCG solver(solver_control);
//#endif
//    LA::MPI::PreconditionAMG preconditioner;
// 
//    LA::MPI::PreconditionAMG::AdditionalData data;
//
//#ifdef USE_PETSC_LA
//      data.symmetric_operator = true;
//#else
//    /* Trilinos defaults are good */
//#endif
//    if (prm.dynamicmode)
//        preconditioner.initialize(system_dynamic_matrix, data);
//    else
//        preconditioner.initialize(system_matrix, data);
//
    if (prm.dynamicmode)
      {
        //PETScWrappers::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
         PETScWrappers::SolverCG cg(solver_control, mpi_communicator);

 
          distributed_locally_relevant_solution = locally_relevant_solution;
          //system_matrix.add(6/(prm.delta_t*prm.delta_t),system_mass_matrix.add(prm.gamma*prm.delta_t,system_damping_matrix.add(prm.beta*prm.delta_t*prm.delta_t,system_matrix)));
          //system_matrix.add(6/(prm.delta_t*prm.delta_t),system_mass_matrix.add(prm.gamma*prm.delta_t,system_damping_matrix));

    
        //TrilinosWrappers::PreconditionSSOR preconditioner;

          PETScWrappers::PreconditionBlockJacobi preconditioner(system_dynamic_matrix);
        //preconditioner.initialize(system_dynamic_matrix);
         // {
         //   TimerOutput::Scope t(computing_timer, "Solve: setup preconditioner");
//
         //   TrilinosWrappers::PreconditionSSOR::AdditionalData additional_data;
         //   preconditioner.initialize(newton_matrix, additional_data);
         // }

            cg.solve(system_dynamic_matrix,
                    distributed_locally_relevant_solution,
                    system_rhs,
                    preconditioner);
          //solver.solve(system_dynamic_matrix,
          //         distributed_locally_relevant_solution,
          //         system_rhs,
          //         preconditioner);
            acceleration_n = distributed_locally_relevant_solution;
            
            constraints_dirichlet_and_hanging_nodes.distribute(distributed_locally_relevant_solution);
            locally_relevant_solution = distributed_locally_relevant_solution;
            pcout << "    Timestepping " << std::endl;

          time_stepping();
      }
    else
    {

          //PETScWrappers::MPI::Vector distributed_locally_relevant_solution(locally_owned_dofs, mpi_communicator);

          PETScWrappers::SolverCG cg(solver_control, mpi_communicator);
          distributed_locally_relevant_solution = locally_relevant_solution;
          PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
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
    }
 

 
    return solver_control.last_step();
  }

  template <int dim>
  void TopLevel<dim>::solve_modes(std::string solver_name, std::string preconditioner_name)
  {

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



    if (this_mpi_process==0)
      std::cout <<"EigenValues Calculation" <<std::endl;

    

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
                                                1e-6,
                                                /*log_history*/ false,
                                                /*log_results*/ false);
    PETScWrappers::SolverCG linear_solver(linear_solver_control);
    linear_solver.initialize(*preconditioner);

    SolverControl solver_control(100,
                                  1e-11,
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
    else
      {
        AssertThrow(false, ExcMessage("not supported eigensolver"));

        eigensolver =
          new dealii::SLEPcWrappers::SolverKrylovSchur(solver_control,
                                                       mpi_communicator);
      }

    // Set the initial vector. This is optional, if not done the initial vector
    // is set to random values
       // PETScWrappers::set_option_value("-pc_factor_mat_solver_typ",
       //                             "mumps");
    eigensolver->set_initial_space(eigenfunctions);

    eigensolver->set_which_eigenpairs(EPS_LARGEST_REAL);
    eigensolver->set_problem_type(EPS_GHEP);

    eigensolver->solve(system_matrix,
                       system_mass_matrix,
                       eigenvalues,
                       eigenfunctions,
                       eigenfunctions.size());
    //eigensolver->set_initial_space(eigenfunctions);
    
     
    //const double               precision = 1e-5;
    //PETScWrappers::MPI::Vector Ax(eigenfunctions[0]), Bx(eigenfunctions[0]);
    //for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
    //  {
    //    system_mass_matrix.vmult(Bx, eigenfunctions[i]);
//
    //    for (unsigned int j = 0; j < eigenfunctions.size(); ++j)
    //      Assert(std::abs(eigenfunctions[j] * Bx - (i == j)) < precision,
    //              ExcMessage("Eigenvectors " + Utilities::int_to_string(i) +
    //                        " and " + Utilities::int_to_string(j) +
    //                        " are not orthonormal!"));
//
    //    system_matrix.vmult(Ax, eigenfunctions[i]);
    //    Ax.add(-1.0 * eigenvalues[i], Bx);
    //    Assert(Ax.l2_norm() < precision,
    //            ExcMessage(Utilities::to_string(Ax.l2_norm())));
    //  }
    //
    //for (auto &eigenfunction : eigenfunctions)
    //    eigenfunction /= eigenfunction.linfty_norm();
  //
//
    delete preconditioner;
    delete eigensolver;
    ////std::cout << std::endl;
    //if (this_mpi_process==0)
    //  for (unsigned int i = 0; i < eigenvalues.size(); ++i)
    //    std::cout << "      Eigenvalue " << i << " : " << eigenvalues[i]
    //              << std::endl;
    //for (auto &eigenfunction : eigenfunctions)
    //  eigenfunction /= eigenfunction.linfty_norm();
   //// return solver_control.last_step();
    //DataOut<dim> data_out_eigen;
 //
    //data_out_eigen.attach_dof_handler(dof_handler);
 //
    //for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
    //        data_out_eigen.add_data_vector(eigenfunctions[i],
    //                           std::string("eigenfunction_") +
    //                             Utilities::int_to_string(i));
  //
    //
 //
    //data_out_eigen.build_patches();
 //
    //std::ofstream output("eigenvectors.vtk");
    //data_out_eigen.write_vtk(output);

  }
 
 
 
 
 
  template <int dim>
  void TopLevel<dim>::output_results() const
  {
    

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
        data_out.add_data_vector(displacement_n, solution_names);
        data_out.add_data_vector(acceleration_n, "acceleration");
        data_out.add_data_vector(velocity_n,"velocity");
        data_out.add_data_vector(displacement_n, "displacement");
      }

    else
      data_out.add_data_vector(locally_relevant_solution, solution_names);

    
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

    //double max_stresses = -1e+20;
    //double max_x_coordinate = -1e+20;
    //std::vector<Vector<double>> center(triangulation.n_active_cells(),Vector<double>(dim));
    Vector<double> center(triangulation.n_active_cells());
    Vector<double> norm_of_stress(triangulation.n_active_cells());
    //std::cout << "Face quardature Formula size=" << GeometryInfo<dim>::faces_per_cell << " with points per cell=" << quadrature_formula.size() << std::endl;
    
    std::vector<Vector<double>> face_index(triangulation.n_active_cells(),Vector<double>(GeometryInfo<dim>::faces_per_cell));
    //Vector<double> max_stress(triangulation.n_active_cells());
  

      
     
      //unsigned int count_faces=0;
      for (auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            // unsigned int count_faces_internal=0;
            const Point<dim> cell_center = cell->center();
            center(cell->active_cell_index())= cell_center(0);
            //center[cell->active_cell_index()](1) = cell_center(1);
            //if (dim == 3)
           //   center[cell->active_cell_index()](2) = cell_center(2);

            SymmetricTensor<2, dim> accumulated_stress;
            for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
              accumulated_stress +=
                reinterpret_cast<PointHistory<dim> *>(cell->user_pointer())[q]
                  .old_stress;
 
            norm_of_stress(cell->active_cell_index()) =
              (accumulated_stress / quadrature_formula.size()).norm();

            
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
    data_out.add_data_vector(center, "center");
    
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
    if (this_mpi_process==0)
    {
      std::filesystem::path dir = "./"+prm.outputfolder;

      // Check if directory does not exist and create it
      if (!std::filesystem::exists(dir))
      {
          std::filesystem::create_directory(dir);
          std::cout << "Directory created\n";
      }
      else
      {
          std::cout << "Directory already exists\n";
      }
    }

    const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(
      "./"+prm.outputfolder+"/", "solution", timestep_no, mpi_communicator, 12);

    //const std::string pvtu_face_filename = data_out_face.write_vtu_with_pvtu_record(
     // "./", "boundary_id",  mpi_communicator, 12);

    data_out_face.write_vtu_in_parallel(
                  prm.outputfolder+"/boundary_id.vtu",mpi_communicator);

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
 
        if (cycle == 0)
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
            solve_modes(prm.eigensolver, prm.eigenprecond);

        solve_timestep();
       

        
        

      }
     
    if (prm.dynamicmode)
      {
        move_mesh(displacement_n);
      }
    else
      {
        move_mesh(locally_relevant_solution);
      }

    {
      TimerOutput::Scope t(computing_timer, "output_results");
      output_results();
    }
 
    pcout << std::endl;
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
    solve_timestep();
 
    if (prm.dynamicmode)
      {
        move_mesh(displacement_n);
      }
    else
      {
        move_mesh(locally_relevant_solution);
      }

    output_results();
    if (present_time == end_time)
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
      triangulation, error_per_cell, 0.4, 0);
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
                vertex_displacement[d] = temp_disp(cell->vertex_dof_index(v, d));

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
    Vector<double> temp_disp;
    temp_disp.reinit(dof_handler.n_dofs());
      
    if (prm.dynamicmode)
      temp_disp=displacement_n;                                            
    else  
      temp_disp=locally_relevant_solution;
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
          //if (prm.dynamicmode)
          //fe_values.get_function_gradients(displacement_n,
          //                                displacement_increment_grads);            
          //else  
          //  fe_values.get_function_gradients(locally_relevant_solution,
          //                                 displacement_increment_grads);
           fe_values.get_function_gradients(temp_disp,
                                         displacement_increment_grads);
          for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
            {
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