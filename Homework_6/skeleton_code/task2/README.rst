Task 2: Inclusive Scan
----------------------

To compile the code run ``make``.

The executable ``inscan`` can be used as follows:

.. code-block::

   # Show all flags.
   ./inscan --help

   # Run all tests and benchmarks.
   ./inscan

   # N <= 32
   ./inscan --warp

   # 32 < N <= 1024
   ./inscan --block

   # 1024 < N <= 1024^2
   ./inscan --medium

   # N > 1024^2
   ./inscan --large

   # For profiling:
   nvprof --print-gpu-trace ./inscan --medium --profile
   nvprof --print-gpu-trace ./inscan --large --profile
