# Runge Kutta 

This is a small collection of a general solver for the Runge Kutta algorithm in different languages.
For the moment, only C/C++ are implemented, with typically the C++ being the first updated with new features. In the future, I plan to also have implementations in native python (which will not be very fast, but useful for quick and dirty tests), Julia and Rust.

The main aim is didactical. However, some care went into the design of the interface, which
can be used reliably for a number of problems. 
At the moment, the algorithm as they are implemented suffer from some limitations (how severe they
are, really depends on what you want from this implementations):
1. They lack an adaptive step size (will probably change in the future)
2. Implicit methods are not supported (will probably never change in the future)
3. Constrained evolution is not supported (if I will implement this, will be only for the case of a vector-like solution, so it will not be terribly general, but it will be probably good enough for most applications)

Implicit methods I will never support because they require a bit too much work to be set up reliably.