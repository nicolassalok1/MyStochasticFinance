from scipy.sparse import diags
from scipy.linalg import lu_factor, lu_solve
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class SecondOrderFDMSolverTemplate(ABC):

    """
    Base class for finding solutions of second order PDE using finite 
    difference method
    """

    def __init__(self,
                 x_min,  # Max value of spatial variable
                 x_max,  # Min value of sptial variable
                 T,  # Total time
                 M,  # Number of time data points
                 N,  # Number of spatial data points
                 # variables only for plotting the solution
                 func_name="u",
                 space_var_name="x",
                 time_var_name="t",
                 terminal_condition_ind=False
                 ):
        self._u = None
        self._x = None
        self._t = None
        self._λ = None
        self._M = M
        self._N = N
        self._δx = None
        self._δt = None
        self._terminal_condition_ind = terminal_condition_ind
        self._solution_viz: SecondOrderFDMSolverTemplate._Solution3DVisualizer = SecondOrderFDMSolverTemplate._Solution3DVisualizer(
            func_name, space_var_name, time_var_name
        )

        self._init_params(x_min, x_max, T)

    def _init_params(self, x_min, x_max, T):
        self._δx = (x_max - x_min) / self._N
        self._δt = T / self._M

        # Create vectors of values of the spatial and time variables.
        # By default a uniform mesh is used for both
        self._x = np.arange(x_min, x_max, self._δx)
        self._t = np.arange(0, T, self._δt)

        self._λ = self._δt / (self._δx * self._δx)
        print("λ " + str(self._λ))

        # Filling up the solution vector with initial
        # & boundary conditions
        self._u = np.zeros((self._N, self._M), dtype="float64")
        if self._terminal_condition_ind:
            self._u[:, self._M-1] = [
                self._initial_condition(x_i) for x_i in self._x]
        else:
            self._u[:, 0] = [
                self._initial_condition(x_i) for x_i in self._x]

        self._u[0, :] = [self._first_boundary_condition(
            t_i) for t_i in self._t]
        self._u[self._N - 1, :] = [
            self._second_boundary_condition(t_i) for t_i in self._t
        ]

    def plot_solution(self):
        self._solution_viz.plot_solution()

    def solve(self):
        """
         Function to iterate over solution vector and populating 
         using stencil computation logic. It may also use other matrix 
         methods than dynamic programming to solve the PDE.
        """
        self._solve_internal()

        self._solution_viz.prepare_solution_for_visual_analysis(
            self._u, self._x, self._t
        )
        return self

    @abstractmethod
    def _solve_internal(self):
        """
        This function should be overriden to include actual computational logic - either dynamic 
        programming or LU decomposition
        """
        ...

    @abstractmethod
    def _initial_condition(self, x):
        ...

    @abstractmethod
    def _first_boundary_condition(self, t):
        ...

    @abstractmethod
    def _second_boundary_condition(self, t):
        ...

    class _Solution3DVisualizer:

        """
        Utiliy class for plotting the solution in 3d
        """

        def __init__(self, func_name="u", space_var_name="x", time_var_name="t"):
            # These variables are needed only for visualisation
            self._func_name = func_name
            self._space_var_name = space_var_name
            self._time_var_name = time_var_name
            self._x_visual = []
            self._t_visual = []
            self._u_visual = []

        def prepare_solution_for_visual_analysis(self, u, x, t):
            n = len(x)
            m = len(t)
            for i in range(n):
                for j in range(m):
                    self._x_visual.append(x[i])
                    self._t_visual.append(t[j])
                    self._u_visual.append(u[i][j])

        def plot_solution(self):
            plt.style.use("seaborn-v0_8")
            func_label = (
                self._func_name
                + "("
                + self._space_var_name
                + ","
                + self._time_var_name
                + ")"
            )
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_trisurf(

                np.asarray(self._x_visual, dtype=np.float64),
                np.asarray(self._t_visual, dtype=np.float64),
                np.asarray(self._u_visual, dtype=np.float64),
                cmap=plt.cm.viridis,
                linewidth=0.2,
                zorder=1,
            )
            ax.set_xlabel(self._space_var_name)
            ax.set_ylabel(self._time_var_name)
#            ax.set_ylabel(self._space_var_name)
#            ax.set_xlabel(self._time_var_name)
            ax.set_zlabel(func_label)
            ax.set_title(func_label + " solution")
            fig.tight_layout()
            plt.show()


class HeatEquationFDMSolverTemplate(SecondOrderFDMSolverTemplate, ABC):

    @abstractmethod
    def _a(self, i=None, j=None):
        ...

    @abstractmethod
    def _b(self, i=None, j=None):
        ...

    @abstractmethod
    def _c(self, i=None, j=None):
        ...


class HeatEquationExplicitFDMSolverTemplate(HeatEquationFDMSolverTemplate, ABC):

    """
     Template Class to solve heat equation using explicit FDM. It 
     doesn't contain the initial & boundary conditions
    """

    def _a(self, i=None, j=None):
        return self._λ

    def _b(self, i=None, j=None):
        return 1 - 2 * self._λ

    def _c(self, i=None, j=None):
        return self._λ

    def _solve_internal(self):
        for i in range(1, self._N - 1):
            for j in range(0, self._M - 1):
                self._u[i][j + 1] = (
                    (self._a(i, j) * self._u[i - 1][j])
                    + (self._b(i, j) * self._u[i][j])
                    + (self._c(i, j) * self._u[i + 1][j])
                )


class HeatEquationExplicitFDMSolver(HeatEquationExplicitFDMSolverTemplate):

    """
     Class to solve heat equation using explicit FDM. It 
     doesn't contain the initial & boundary conditions
    """

    def _initial_condition(self, x):
        return np.sin(2*np.pi*x)

    def _first_boundary_condition(self, t):
        return 0

    def _second_boundary_condition(self, t):
        return 0


class HeatEquationImplicitFDMSolverTemplate(HeatEquationFDMSolverTemplate, ABC):

    """
     Abstract class to solve heat equation using implicit FDM. It doesn't contain 
     initial & boundary conditions
    """

    def _a(self, i=None, j=None):
        return -self._λ

    def _b(self, i=None, j=None):
        return 2*self._λ + 1

    def _c(self, i=None, j=None):
        return -self._λ

    def _solve_internal(self):
        a = [self._a(i=i) for i in range(self._N)]
        b = [self._b(i=i) for i in range(self._N)]
        c = [self._c(i=i) for i in range(self._N)]

        '''
          Create a tridiagonal matrix of coefficients represented 
          by A in equation
              〖Au〗_((j+1))= u_((j) )+ b_((j) )
        '''
        A = diags([a, b, c],
                  [-1, 0, 1], shape=(self._N-1, self._N-1), dtype="float64").toarray()

        '''
           Decompose A into lower-triangular L & upper-triangular
           U by LU factorization
        '''
        lu, piv = lu_factor(A)

        '''
          Initialize an array β for boundary values represented 
          by b in equation
              〖Au〗_((j+1))= u_((j) )+ b_((j) )
        '''
        β = np.zeros(self._N-1, dtype="float64")
        from_index, to_index = (
            self._M-2, 0) if self._terminal_condition_ind else (self._M-1, 1)

        for j in range(from_index, to_index, -1):
            β[0] = a[1] * self._u[0][j]
            β[-1] = c[-1] * self._u[-1][j]

            '''
              Directly get the solution by lu_solve and
              avoid computing 
                   u_((j+1))=U^(-1) {L^(-1) (u_((j) )+ b_((j) ) )}
              which is a two step process
            '''
            self._u[1:, j-1] = lu_solve((lu, piv), self._u[1:, j] + β)


class HeatEquationImplicitFDMSolver(HeatEquationImplicitFDMSolverTemplate):

    """
     Class to solve heat equation using implicit FDM. It contains 
     initial & boundary conditions
    """

    def _initial_condition(self, x):
        return np.sin(2*np.pi*x)

    def _first_boundary_condition(self, t):
        return 0

    def _second_boundary_condition(self, t):
        return 0


class BaseHeatEquationCrankNicolsonFDMSolverTemplate(HeatEquationFDMSolverTemplate, ABC):

    @abstractmethod
    def _d(self, i=None, j=None):
        ...


class HeatEquationCrankNicolsonFDMSolverTemplate(BaseHeatEquationCrankNicolsonFDMSolverTemplate, ABC):

    """
     Abstract class to solve heat equation using Crank-Nicolson FDM. It It doesn't contain  
     initial & boundary conditions
    """

    def _a(self, i=None, j=None):
        return self._λ

    def _b(self, i=None, j=None):
        return 2 + 2*self._λ

    def _c(self, i=None, j=None):
        return self._λ

    def _d(self, i=None, j=None):
        return 2 - 2*self._λ

    def _solve_internal(self):
        a = np.array([self._a(i=i) for i in range(self._N-1)])
        b = np.array([self._b(i=i) for i in range(self._N-1)])
        c = np.array([self._c(i=i) for i in range(self._N-1)])
        d = np.array([self._d(i=i) for i in range(self._N-1)])

        '''
          Create a tridiagonal matrix of coefficients represented 
          by A in equation
              〖Au〗_((j+1))= Bu_((j) )+ b_((j) )
        '''
        A = diags([-a, b, -c],
                  [-1, 0, 1], shape=(self._N-1, self._N-1), dtype="float64").toarray()

        '''
          Create a tridiagonal matrix of coefficients represented 
          by B in equation
              〖Au〗_((j+1))= Bu_((j) )+ b_((j) )
        '''
        B = diags([a, d, c],
                  [-1, 0, 1], shape=(self._N-1, self._N-1), dtype="float64").toarray()

        lu, piv = lu_factor(A)

        '''
          Initialize an array β for boundary values represented 
          by b in equation
              〖Au〗_((j+1))= Bu_((j) )+ b_((j) )
        '''
        β = np.zeros(self._N-1, dtype="float64")

        from_index, to_index = (
            self._M-2, 0) if self._terminal_condition_ind else (self._M-1, 1)

        for j in range(from_index, to_index, -1):
            β[0] = a[1] * (self._u[0][j] + self._u[0][j-1])
            β[-1] = c[-1] * (self._u[-1][j] + self._u[-1][j-1])

            '''
              Directly get the solution by lu_solve and
              avoid computing 
                   u_((j+1))=U^(-1) {L^(-1) (〖Bu〗_((j) )+ b_((j) ) )}
              which is a two step process
            '''
            self._u[1:, j-1] = lu_solve((lu, piv),
                                        B.dot(self._u[1:, j]) + β)


class HeatEquationCrankNicolsonFDMSolver(HeatEquationCrankNicolsonFDMSolverTemplate):

    """
     Class to solve heat equation using Crank-Nicolson FDM. It contains 
     initial & boundary conditions
    """

    def _initial_condition(self, x):
        return np.sin(2*np.pi*x)

    def _first_boundary_condition(self, t):
        return 0

    def _second_boundary_condition(self, t):
        return 0
