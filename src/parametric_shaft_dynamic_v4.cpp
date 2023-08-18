#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
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
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_postprocessor.h> 
#include <deal.II/base/index_set.h>

#include <deal.II/matrix_free/fe_evaluation_data.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/vector.h>

#include <deal.II/base/symmetric_tensor.h>
 #include <deal.II/base/parameter_handler.h>

#include <deal.II/physics/transformations.h>

#include <deal.II/lac/slepc_solver.h>

#include <deal.II/base/hdf5.h>
#include "petscmat.h" 


#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
 
#include <cstdlib> 


 const double pi = 3.14159265358979323846;
const double rpm_to_rps = pi/30; // Conversion factor from RPM to RPS
namespace Step_18_cyl
{
  using namespace dealii;
 
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
        prm.declare_entry("Viscocity",
                          "0.001",
                          Patterns::Double(0.0),
                          "Viscocity of the bearing");
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
      }
      prm.leave_subsection();
    }

    struct Solver
    {
      double tol;
      unsigned int max_iter;
      unsigned int n_refinement_cycles;
      bool eigenmodes;
      std::string eigenmodenumber;
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
        eigenmodenumber = prm.get("EigenmodesNumber");
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
  };
 
 
 
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
    
    
    // FOR ALL TIMESTEPS
    void assemble_system();

    void time_stepping();
 
    void solve_timestep();
 
    unsigned int solve_linear_problem();
    void solve_modes(std::string solver_name, std::string preconditioner_name);
 
    void output_results() const;
  
    void do_initial_timestep(); //for first timestep ONLY
 
    void do_timestep();  //for ALL timesteps
 
    void refine_initial_grid(); //for first timestep ONLY
 
    
    void move_mesh();   //for ALL timesteps
 
    void setup_quadrature_point_history(); // for 1st timestep to et up a pristine state for the history variables, ONLY IF the quadrature points on cells belong to the present processor
 
    void update_quadrature_point_history();


    Parameters::AllParameters prm;

 
    parallel::shared::Triangulation<dim> triangulation;

    FESystem<dim> fe;
 
    DoFHandler<dim> dof_handler;
 
    AffineConstraints<double> hanging_node_constraints;
 
    const QGauss<dim> quadrature_formula;
    const QGauss<dim-1> face_quadrature_formula;

 
    std::vector<PointHistory<dim>> quadrature_point_history; // in step_4 will use CellDataStorage, in this manually
 
 
 
    PETScWrappers::MPI::SparseMatrix system_matrix;
    PETScWrappers::MPI::SparseMatrix system_mass_matrix;
    PETScWrappers::MPI::SparseMatrix system_stiffness_matrix;
    PETScWrappers::MPI::SparseMatrix system_damping_matrix;



    PETScWrappers::MPI::SparseMatrix cell_stiffness_matrix;
    PETScWrappers::MPI::Vector system_rhs;
    Vector<double> displacement_n;
    Vector<double> velocity_n;
    Vector<double> acceleration_n;



    std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
    std::vector<PetscScalar>                eigenvalues;
    
    //AffineConstraints<double> constraints4modes;


    Vector<double> incremental_displacement;
    //Vector<double> velocity_n;
 
    
    double       present_time;
    double       present_timestep;
    double       end_time;
    unsigned int timestep_no;


    // FOR PARALLEL PROCESSING
    MPI_Comm mpi_communicator;
 
    const unsigned int n_mpi_processes;
 
    const unsigned int this_mpi_process;
 
    ConditionalOStream pcout;
 

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
      PropellerTorque(const double torque, const double distance2centroid, const double angle4force, const double polar_moment);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override;
      virtual void 
      vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  value_list) const override;
      
    private:
      const double torque; 
      const double distance2centroid;
      const double angle4force;
      const double polar_moment;
  };

  template <int dim>
  PropellerTorque<dim>::PropellerTorque(const double torque, const double distance2centroid, const double angle4force, const double polar_moment)
    : Function<dim>(dim), torque(torque), distance2centroid(distance2centroid), angle4force(angle4force), polar_moment(polar_moment)
  {}

  template <int dim>
  inline void PropellerTorque<dim>::vector_value(const Point<dim> &/*p*/,
                                                Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);
    double tau = torque*distance2centroid/(polar_moment);
    values = 0;
    values(1) = tau* cos(angle4force);
    values(2) = tau* sin(angle4force);

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

  template <int dim>
  TopLevel<dim>::TopLevel(const std::string &input_file)
    : prm(input_file)
    , triangulation(MPI_COMM_WORLD)
    , fe(FE_Q<dim>(1), dim)
    , dof_handler(triangulation)
    , quadrature_formula(fe.degree + 1)
    //, face_quadrature_formula(fe_values.get_fe().degree + 1)
    , face_quadrature_formula(fe.degree + 1)
    , present_time(0)
    , present_timestep(prm.delta_t)
    , end_time(prm.end_time)
    , timestep_no(0)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , pcout(std::cout, this_mpi_process == 0)
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
      do_timestep();
  }
 
 
 //TODO CHECK THE BOUNDARIES

  template <int dim>
  void TopLevel<dim>::create_coarse_grid()
  {
    const double radius = prm.radius;
    const double half_length = prm.half_length;
    const double bearing_x = prm.bearing_x;
    const double bearing_length = prm.bearing_length;
    GridGenerator::subdivided_cylinder(triangulation, prm.partitions, radius, half_length);

    triangulation.refine_global(prm.n_global_refinements);
    //const double x0 = -prm.half_length;
    //const double x1 =  prm.half_length;
    //const double dL = (x1 - x0) / n_mpi_processes;
    GridTools::partition_triangulation (n_mpi_processes,
                                           triangulation);
  
    //typename DoFHandler<dim>::active_cell_iterator
     // cell = dof_handler.begin_active(), endc = triangulation.end();
    
   /* / for (const auto &cell : triangulation.active_cell_iterators())
      {
        const dealii::Point<dim> &center = cell->center();
        const double              x      = center[0];

        const auto id = static_cast<unsigned int>((x - x0) / dL);
        cell->set_subdomain_id(id);
      }*/
//for (; cell != endc; ++cell)
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
          {
            const Point<dim> face_center = face->center();
 
            if (face_center[0] == half_length)
              face->set_boundary_id(0);
            else if (face_center[0] == -half_length)
              face->set_boundary_id(1);
     
            else if (std::sqrt(face_center[1] * face_center[1] +
                              face_center[2] * face_center[2]) <
                    radius)
                   if (face_center[0] <=(bearing_x+bearing_length/2) && face_center[0]>=(bearing_x-bearing_length/2))
                    face->set_boundary_id(3);  
                  else    
                    face->set_boundary_id(2);
            
                                //! BEARING BELOW   
            else
              face->set_boundary_id(4);
          }

    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary()==false)
          face->set_boundary_id(5);


        
          

    setup_quadrature_point_history();
  }
 
 
 
  template <int dim>
  void TopLevel<dim>::setup_system()
  {
    /*
    const double x0 = -prm.half_length;
    const double x1 =  prm.half_length;
    const double dL = (x1 - x0) / n_mpi_processes;

    active_cell_iterator cell = triangulation.begin_active(),
                                                  endc = triangulation.end();
    for (; cell != endc; ++cell)
      {
        const dealii::Point<dim> &center = cell->center();
        const double              x      = center[0];

        const auto id = static_cast<unsigned int>((x - x0) / dL);
        cell->set_subdomain_id(id);
      }
    */
    
    //distributes the dofs to the processors
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::subdomain_wise(dof_handler);
    std::vector<dealii::IndexSet> locally_owned_dofs_per_processor =
        DoFTools::locally_owned_dofs_per_subdomain(dof_handler);

    locally_owned_dofs = locally_owned_dofs_per_processor[this_mpi_process];
    //locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
 

    
    //making the hanging nodes into constraints to preserve the 
    //linearity of the global function on the boundary of each cell
    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();


    // Creating the Sparsity of the Matrix
    //DynamicSparsity instead of SparsityPattern 
    //because the simple needs the initial upper bound for the # of entries in each row w/ DoFHandler::max_couplings_between_dofs()
    //which in 3D increases a lot the allocated memory initially(100s of MBs)

    DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    sparsity_pattern,
                                    hanging_node_constraints,
                                    /*keep constrained dofs*/ false);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    std::vector<dealii::types::global_dof_index> n_locally_owned_dofs(n_mpi_processes);

    for (unsigned int i = 0; i < n_mpi_processes; ++i)
      n_locally_owned_dofs[i] = locally_owned_dofs_per_processor[i].n_elements();


    SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                               n_locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);
    //PETSc preallocating the system matrix
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         sparsity_pattern,
                         mpi_communicator);
    
    //PETSc correcting the preallocated size of the  righthand side and dx matrices


    
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    incremental_displacement.reinit(dof_handler.n_dofs());
    displacement_n.reinit(dof_handler.n_dofs());
    velocity_n.reinit(dof_handler.n_dofs());
    acceleration_n.reinit(dof_handler.n_dofs());

    displacement_n = 0;
    velocity_n = 0;
    acceleration_n = 0;
  

    {
    VectorTools::interpolate_boundary_values(
      dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
    
    constraints.close();

    DynamicSparsityPattern csp(locally_relevant_dofs);
    // Fill in ignoring all cells that are not locally owned
    DoFTools::make_sparsity_pattern(dof_handler,
                                    csp,
                                    constraints,
                                    /*keep constrained dofs*/ true);


    SparsityTools::distribute_sparsity_pattern(csp,
                                                     n_locally_owned_dofs,
                                                     mpi_communicator,
                                                     locally_relevant_dofs); 
    }
    system_mass_matrix.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          sparsity_pattern,
                          mpi_communicator);
    
    system_stiffness_matrix.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          sparsity_pattern,
                          mpi_communicator);
    system_damping_matrix.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          sparsity_pattern,
                          mpi_communicator);
    
    

    eigenfunctions.resize(dof_handler.n_dofs());
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
    
    //std::cout << "\n System Setup Complete"<<std::endl;
  }
 
 
 
 
  template <int dim>
  void TopLevel<dim>::assemble_system()
  {
    system_rhs    = 0;
    system_matrix = 0;
    system_mass_matrix = 0;
    system_stiffness_matrix = 0;
    system_damping_matrix = 0;
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
    




    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
    BodyForce<dim>              body_force(prm.rho, prm.g);
    std::vector<Vector<double>> body_force_values(n_q_points,
                                                  Vector<double>(dim));


    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell ;
    
    double total_area = 0.0;
    Point<dim> weighted_position;
   // Vector<double> boundary_1();

    typename dealii::DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    //std::cout << "\n Assembly constants and matrices INITIALISED"<<std::endl;
    //for (const auto &cell : dof_handler.active_cell_iterators())
    //if (this_mpi_process==0)
    for (; cell != endc; ++cell)
      //if (cell->is_locally_owned())
     // if (cell->subdomain_id() == this_mpi_process)
        {
            if (cell->at_boundary())
            {
                for (unsigned int face = 0; face < n_faces_per_cell; ++face)
                {
                    if (cell->face(face)->boundary_id() == 1)
                    {
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


    const double total_force = rpmToForce(rpm, prm.MCR/pow(prm.rpm_MCR*rpm_to_rps,3), prm.v_ship);
    const double total_torque = rpmToTorque(rpm,prm.MCR/pow(prm.rpm_MCR*rpm_to_rps,3));
    Point<dim> centroid_;
    const double pressure = total_force / total_area;
    Vector<double> pressure_vector(dofs_per_cell);
    //pressure_vector(0) = pressure;
    Vector<double> distance2centroid(n_q_points);
    Vector<double> angle4force(n_q_points);
    //pressure_vector(0) = pressure;


    PropellerForce<dim> propeller_pressure(pressure); 
    std::vector<Vector<double>> propeller_pressure_values(n_q_points,
                                                  Vector<double>(dim));
    
    std::vector<Vector<double>> propeller_pressure_dofs(n_q_points,
                                                  Vector<double>(dim));




    if(total_area > 0.0)
    {
        centroid_ = weighted_position / total_area;
    }

    double polar_moment = 0.0;
    cell = dof_handler.begin_active();
    //for (const auto &cell : dof_handler.active_cell_iterators())
      //  if (cell->is_locally_owned())
    //if (this_mpi_process==0)
    for (; cell != endc; ++cell)
      //if (cell->is_locally_owned())
     // if (cell->subdomain_id() == this_mpi_process)
        {
            if (cell->at_boundary())
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
                            distance2centroid[q]= relative_position.norm();
                            angle4force[q] = atan2(relative_position(1),relative_position(0));
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

    if (this_mpi_process==0)
      std::cout <<"\n Total Area of section: "<< total_area << "\n Pressure=  " << pressure <<" for RPM="<<prm.rpm << "\n Centroid=  " << centroid_ << "\n Polar Moment of Inertia J=  " << polar_moment<< std::endl;

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

    //for (const auto &cell : dof_handler.active_cell_iterators()) 
      //if (cell->is_locally_owned())
    unsigned int count=0;
    unsigned int countboundary1=0;
    cell = dof_handler.begin_active();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
      //if (cell->subdomain_id() == this_mpi_process)
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;

          cell_stiffness_matrix = 0;
          cell_mass_matrix      = 0;
          cell_damping_matrix   = 0;

          
         
          
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

                  cell_stiffness_matrix(i, j) +=     (cell_matrix(i, j));     

                  
                  cell_mass_matrix(i, j) += (prm.rho * prm.g *
                                            fe_values.shape_value(i, q_point) *
                                            fe_values.shape_value(j, q_point) *
                                            fe_values.JxW(q_point));
                  cell_damping_matrix(i, j) +=  prm.mu_rayleigh*cell_mass_matrix(i, j) + prm.lambda_rayleigh*cell_matrix(i, j);
                }

          // local_quadrature_points_data is a pointer for PointHistory value when used with *
          
          const PointHistory<dim> *local_quadrature_points_data =
            reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
          body_force.vector_value_list(fe_values.get_quadrature_points(),
                                       body_force_values);


          
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                fe.system_to_component_index(i).first;
 
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                 
                  const SymmetricTensor<2, dim> &old_stress =
                    local_quadrature_points_data[q_point].old_stress;
                    cell_rhs(i) +=
                      ((body_force_values[q_point](component_i)) *
                        fe_values.shape_value(i, q_point) -
                      old_stress * get_strain(fe_values, i, q_point)) *
                      fe_values.JxW(q_point);

                }
            }
         propeller_pressure.vector_value_list(fe_values.get_quadrature_points(), 
                                       propeller_pressure_values);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
                            const unsigned int component_i =
                                        fe.system_to_component_index(i).first;
                      
          for (unsigned int face = 0; face < n_faces_per_cell; ++face)
          if (cell->face(face)->boundary_id() == 1)
          //if (cell->boundary_id() == 1)
              {
                double rhs_value=0;
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {

                    Point<dim> q_point_point = fe_values.quadrature_point(q_point);
                    Point<dim> relative_position(q_point_point - centroid_);
                    distance2centroid[q_point]= relative_position.norm();
                    angle4force[q_point] = atan2(relative_position(1),relative_position(0));
                    PropellerTorque<dim> propeller_torque(total_torque,distance2centroid[q_point], angle4force[q_point], polar_moment); 
                    std::vector<Vector<double>> propeller_torque_values(n_q_points,
                                                                  Vector<double>(dim));
                    
                    std::vector<Vector<double>> propeller_torque_dofs(n_q_points,
                                                                  Vector<double>(dim));

                    propeller_torque.vector_value_list(fe_values.get_quadrature_points(), 
                                       propeller_torque_values);

                      if (prm.dynamicmode)
                          {
                              double velocity_q = velocity_n[cell->vertex_dof_index(q_point,0)];
                              double accel_q = acceleration_n[cell->vertex_dof_index(q_point,0)];

                              rhs_value += (fe_values.shape_value(i,q_point) *
                                        (velocity_q +
                                        prm.delta_t*(1-prm.gamma)*accel_q) +
                                        prm.delta_t*prm.delta_t*(0.5-prm.beta)*fe_values.shape_value(i,q_point) *
                                        accel_q) *
                                      fe_values.JxW(q_point);
                          }
                      else
                          rhs_value += 0;

                

                      cell_rhs(i) += ((propeller_pressure_values[q_point](component_i)+propeller_torque_values[q_point](component_i))*
                                      fe_values.shape_value(i, q_point) ) *
                                      fe_values.JxW(q_point)+rhs_value;
                      countboundary1 +=1;

                      if ((cell_rhs(i))>1e70)
                      {
                        std::cout << "pressure_vector(i) = " << propeller_pressure_values[q_point](component_i) << std::endl;
                        std::cout << "torquevector(i) = " << propeller_torque_values[q_point](component_i) << std::endl;
                        //std::cout << "fe_face_values.shape_value(i, q_point) = " << fe_face_values.shape_value(i, q_point) << std::endl;
                        std::cout << "fe_values.JxW(q_point) = " << fe_values.JxW(q_point) << std::endl;
                      }
                      else if (propeller_pressure_values[q_point](component_i)<pressure)
                      {
                        //std::cout << "pressure=0"<< std::endl;
                        count +=1;
                      }
                    }
              }
          }
              
                        
                  
              
          
        
          cell->get_dof_indices(local_dof_indices);
          
          hanging_node_constraints.distribute_local_to_global(cell_matrix,
                                                              cell_rhs,
                                                              local_dof_indices,
                                                              system_matrix,
                                                              system_rhs);

          constraints.distribute_local_to_global(cell_stiffness_matrix,
                                               local_dof_indices,
                                               system_stiffness_matrix);
          constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               system_mass_matrix);     
          constraints.distribute_local_to_global(cell_damping_matrix,
                                                local_dof_indices,
                                                system_damping_matrix);

        
        }
    //if (count==countboundary1) 
     //   std::cout << "pressure=0 for "<< this_mpi_process << std::endl;
     //else
     //  std::cout << "pressure=0 for "<< count <<" in "<<countboundary1 <<std::endl;

    system_matrix.compress(VectorOperation::add);
    system_mass_matrix.compress(VectorOperation::add);
    system_stiffness_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    system_damping_matrix.compress(VectorOperation::add);
    //TODO fix rayleigh damping matrix

    //system_damping_matrix = (0.025*=(system_mass_matrix))+(0.023*=(system_matrix));
 
 /*   double min_spurious_eigenvalue = std::numeric_limits<double>::max(),
     max_spurious_eigenvalue = -std::numeric_limits<double>::max();
 
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      if (constraints.is_constrained(i))
        {
          const double ev         = system_stiffness_matrix(i, i) / system_mass_matrix(i, i);
          min_spurious_eigenvalue = std::min(min_spurious_eigenvalue, ev);
          max_spurious_eigenvalue = std::max(max_spurious_eigenvalue, ev);
        }
 
    std::cout << "   Spurious eigenvalues are all in the interval " << '['
              << min_spurious_eigenvalue << ',' << max_spurious_eigenvalue
              << ']' << std::endl;*/


 
    const FEValuesExtractors::Scalar          z_component(dim-1);
    //const FEValuesExtractors::Scalar          bearing_component(dim-1);

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(dim),
                                             boundary_values);   
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
    VectorTools::interpolate_boundary_values(dof_handler,
                                                3,
                                                IncrementalBoundaryValues<dim>(prm.end_time, present_timestep,prm.direction,prm.displacement),
                                                boundary_values,
                                                fe.component_mask(z_component)); 
 
    PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, tmp, system_rhs, false);
    incremental_displacement = tmp;
    //pcout << "    Assembiling System for " << this_mpi_process <<" process done." <<std::endl;
  }
 
 
 
  template <int dim>
  void TopLevel<dim>::time_stepping()
  {
    Vector<double> displacement_np(dof_handler.n_dofs());  // Displacement at time n+1
    Vector<double> velocity_np(dof_handler.n_dofs());  // Velocity at time n+1```cpp
    Vector<double> acceleration_np(dof_handler.n_dofs());  // Acceleration at time n+1
    Vector<double> velocity_np_1(dof_handler.n_dofs());
    Vector<double> velocity_np_2(dof_handler.n_dofs());   
    Vector<double> velocity_np_3(dof_handler.n_dofs()); 
    // Solve for displacement_np
    displacement_np = incremental_displacement;

    // Update velocity and acceleration with Newmark-beta method
    velocity_np_1 = velocity_n;
    velocity_np_2 = acceleration_n;
    velocity_np_2 *=(1-prm.gamma);
    velocity_np_3 = acceleration_np;
    velocity_np_3 *= (prm.gamma);

    velocity_np.add(1.0,velocity_np_2);
    velocity_np.add(1.0,velocity_np_3);//
    velocity_np *= prm.delta_t;
    velocity_np.add(1.0,velocity_np,1.0,velocity_np_1);
    acceleration_np = acceleration_n;
    velocity_np_1 = displacement_np;
    velocity_np_1.add(-1,displacement_n);
    acceleration_np.add(1/(prm.beta*prm.delta_t*prm.delta_t),velocity_np_1);
    acceleration_np.add(-1/(prm.beta*prm.delta_t),velocity_n);
    acceleration_np.add(-(1/(2*prm.beta)-1),acceleration_n);
    
    // Update the vectors for the next time step
    displacement_n = displacement_np;
    velocity_n = velocity_np;
    acceleration_n = acceleration_np;
  }

  template <int dim>
  void TopLevel<dim>::solve_timestep()
  {
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
    PETScWrappers::MPI::Vector distributed_incremental_displacement(
      locally_owned_dofs, mpi_communicator);
    distributed_incremental_displacement = incremental_displacement;
    
    SolverControl solver_control(dof_handler.n_dofs(),
                                 prm.tol * system_rhs.l2_norm());
 
    PETScWrappers::SolverCG cg(solver_control, mpi_communicator);
 
   
    if (prm.dynamicmode)
          //system_matrix.add(6/(prm.delta_t*prm.delta_t),system_mass_matrix.add(prm.gamma*prm.delta_t,system_damping_matrix.add(prm.beta*prm.delta_t*prm.delta_t,system_matrix)));
          system_matrix.add(6/(prm.delta_t*prm.delta_t),system_mass_matrix.add(prm.gamma*prm.delta_t,system_damping_matrix));

    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
    cg.solve(system_matrix,
             distributed_incremental_displacement,
             system_rhs,
             preconditioner);
    if (prm.dynamicmode)
      time_stepping();
    incremental_displacement = distributed_incremental_displacement;
 
    hanging_node_constraints.distribute(incremental_displacement);
 
    return solver_control.last_step();
  }

  template <int dim>
  void TopLevel<dim>::solve_modes(std::string solver_name, std::string preconditioner_name)
  {
  
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

    dealii::SolverControl   linear_solver_control(dof_handler.n_dofs(),
                                                1e-15,
                                                /*log_history*/ false,
                                                /*log_results*/ false);
    PETScWrappers::SolverCG linear_solver(linear_solver_control);
    linear_solver.initialize(*preconditioner);

    SolverControl solver_control(prm.max_iter,
                                         1e-12);

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

        // Make compiler happy and not complaining about non
        // uninitialized variables
        eigensolver =
          new dealii::SLEPcWrappers::SolverKrylovSchur(solver_control,
                                                       mpi_communicator);
      }

    // Set the initial vector. This is optional, if not done the initial vector
    // is set to random values
    eigensolver->set_initial_space(eigenfunctions);

    eigensolver->set_which_eigenpairs(EPS_LARGEST_REAL);
    eigensolver->set_problem_type(EPS_HEP);

    eigensolver->solve(system_stiffness_matrix,
                       eigenvalues,
                       eigenfunctions,
                       eigenfunctions.size());
    /*SolverControl eigen_solver_control(dof_handler.n_dofs(), 1e-9);

    SLEPcWrappers::SolverKrylovSchur eigensolver(eigen_solver_control, mpi_communicator);

    eigensolver.set_which_eigenpairs(EPS_SMALLEST_REAL);
 
    eigensolver.set_problem_type(EPS_GHEP);

    //PETScWrappers::SolverCG cgmodes(eigen_solver_control, mpi_communicator);
 
    PETScWrappers::PreconditionBlockJacobi preconditionermodes(system_stiffness_matrix);

    cgmodes.solve(system_stiffness_matrix,
                      system_mass_matrix,
                      eigenvalues,
                      eigenfunctions,
                      eigenfunctions.size());
    eigensolver.solve(system_stiffness_matrix,
                      system_mass_matrix,
                      eigenvalues,
                      eigenfunctions,
                      eigenfunctions.size());*/
    //for (unsigned int i = 0; i < eigenvalues.size(); ++i)
   //   dealii::deallog << eigenvalues[i] << std::endl;

    delete preconditioner;
    delete eigensolver;
    //std::cout << std::endl;
    if (this_mpi_process==0)
      for (unsigned int i = 0; i < eigenvalues.size(); ++i)
        std::cout << "      Eigenvalue " << i << " : " << eigenvalues[i]
                  << std::endl;
   /* DataOut<dim> data_out_eigen;
 
    data_out_eigen.attach_dof_handler(dof_handler);
 
    for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
data_out_eigen.add_data_vector(eigenfunctions[i],
                               std::string("eigenfunction_") +
                                 Utilities::int_to_string(i));
    for (unsigned int i = 0; i < eigenvalues.size(); ++i)
      std::cout << "      Eigenvalue " << i << " : " << eigenvalues[i]
                << std::endl;
    
 
    data_out_eigen.build_patches();
 
   std::ofstream output("eigenvectors.vtk");
    data_out_eigen.write_vtk(output);*/

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
 
    data_out.add_data_vector(incremental_displacement, solution_names);

    
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
        
        solve_timestep();
       

        
        

      }
      if (prm.eigenmodes)
        solve_modes(prm.eigensolver, prm.eigenprecond);
 
    move_mesh();
    output_results();
 
    pcout << std::endl;
  }
 
 
 
 
  template <int dim>
  void TopLevel<dim>::do_timestep()
  {
    present_time += present_timestep;
    ++timestep_no;
    pcout << "Timestep " << timestep_no << " at time " << present_time
          << std::endl;

    if (present_time > end_time)
      {
        present_timestep = -(present_time - end_time);
        present_time = end_time;
        

      }
 
 
    solve_timestep();
 
    move_mesh();
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
    Vector<float> error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      incremental_displacement,
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
    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    error_per_cell,
                                                    0.1,
                                                    0.);
    triangulation.execute_coarsening_and_refinement();
 
    setup_quadrature_point_history();
  }
 
 
 
 
  template <int dim>
  void TopLevel<dim>::move_mesh()
  {
    pcout << "    Moving mesh..." << std::endl;
 
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);

    //for (auto &cell : dof_handler.active_cell_iterators())
    typename dealii::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    for (; cell != endc; ++cell)
      for (const auto v : cell->vertex_indices())
        if (vertex_touched[cell->vertex_index(v)] == false)
          {
            vertex_touched[cell->vertex_index(v)] = true;
 
            Point<dim> vertex_displacement;
            for (unsigned int d = 0; d < dim; ++d)
              vertex_displacement[d] =
                incremental_displacement(cell->vertex_dof_index(v, d));
 
            cell->vertex(v) += vertex_displacement;
          }
  }
 
 
 
  template <int dim>
  void TopLevel<dim>::setup_quadrature_point_history()
  {
 
    //triangulation.clear_user_data();
 
    {
      std::vector<PointHistory<dim>> tmp;
      quadrature_point_history.swap(tmp);
    }
    quadrature_point_history.resize(
      triangulation.n_locally_owned_active_cells() * quadrature_formula.size());
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    unsigned int history_index = 0;
    //for (auto &cell : triangulation.active_cell_iterators())
      //if (cell->is_locally_owned())
     for (; cell != endc; ++cell)
      if (cell->subdomain_id() == this_mpi_process)
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
    auto stress_strain_tensor_ = get_stress_strain_tensor<dim>(prm.lambda, prm.mu);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients);
 
    std::vector<std::vector<Tensor<1, dim>>> displacement_increment_grads(
      quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));
 
    for (auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          PointHistory<dim> *local_quadrature_points_history =
            reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
          Assert(local_quadrature_points_history >=
                   &quadrature_point_history.front(),
                 ExcInternalError());
          Assert(local_quadrature_points_history <=
                   &quadrature_point_history.back(),
                 ExcInternalError());
 
          fe_values.reinit(cell);
          fe_values.get_function_gradients(incremental_displacement,
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
      //using namespace dealii;
      using namespace Step_18_cyl;
 
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