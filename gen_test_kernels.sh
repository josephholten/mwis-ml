for graph in /home/jholten/test_graphs/*.graph; do
    find /home/jholten/kernels/kamis_results -wholename "$graph.kernel"
done
