"""A module providing numerical solvers for nonlinear equations."""


class ConvergenceError(Exception):
    """Exception raised if a solver fails to converge."""
    pass 

def newton_raphson(f, df, x_0, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using Newton-Raphson iteration.

    Solve f==0 using Newton-Raphson iteration.

    Parameters
    ----------
    f : function(x: float) -> float 
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the iteration.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using Newton iteration.
    """
    for i in range(max_its+1):
        if abs(f(x_0)) < eps:
            return x_0
        x_1 = x_0 - f(x_0)/df(x_0)
        x_0 = x_1 
    raise ConvergenceError("The newton_raphson solver failed to converge !")
    
    # try:
    #     for i in range(max_its+1):
    #         if abs(f(x_0)) < eps:
    #             return x_0
    #         x_1 = x_0 - f(x_0)/df(x_0)
    #         x_0 = x_1 
    #     raise ConvergenceError
    # except ConvergenceError:
    #     print("The newton_raphson solver failed to converge !")
    #     raise ConvergenceError

def bisection(f, x_0, x_1, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using bisection.

    Solve f==0 using bisection starting with the interval [x_0, x_1]. f(x_0)
    and f(x_1) must differ in sign.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    x_0 : float
        The left end of the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using bisection.
    """
    if f(x_0)*f(x_1) > 0:
        raise ValueError("f(x_0) and f(x_1) must differ in sign.")
    
    for i in range(1, max_its+1):
        x_mid = (x_0 + x_1)/2
        if abs(f(x_mid)) < eps:
            return x_mid
        if f(x_0)*f(x_mid) > 0:
            x_0 = x_mid
        else:
            x_1 = x_mid
    raise ConvergenceError("The bisection solver failed to converge !")


def solve(f, df, x_0, x_1, eps=1.0e-5, max_its_n=20, max_its_b=20):
    """Solve a nonlinear equation.

    solve f(x) == 0 using Newton-Raphson iteration, falling back to bisection
    if the former fails.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the Newton-Raphson iteration, and left end of
        the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its_n : int
        The maximum number of iterations to be taken before the newton-raphson
        solver is taken to have failed.
    max_its_b : int
        The maximum number of iterations to be taken before the bisection
        solver is taken to have failed.

    Returns
    -------
    float
        The approximate root.
    """
    try:
        x_n = newton_raphson(f, df, x_0, eps, max_its_n)
    except ConvergenceError:
        print("Newton-Raphson method failed to converge. Now trying Bisection method")
        x_b = bisection(f, x_0, x_1, eps, max_its_b)
    else:
        return x_n
    
    return x_b
