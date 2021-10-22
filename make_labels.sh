FOLDER=/home/joseph/uni/sem5/christian/mwis-ml

for graph in $FOLDER/instances/*.graph; do
    b=$(basename $graph)
    m=$FOLDER/instances/${b%.*}.mis
    if [ ! -e $m ]; then  # mis does not exist
        echo "$graph"
        $FOLDER/weighted_branch_reduce $graph --out=$m
    fi
done





