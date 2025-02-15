for (( i=1; i<=5; i++ ))
do
    num1=$[ $RANDOM % 5005 + 1 ]
    num2=$[ $RANDOM % 10086 + 1 ]
    echo $num1 $num2
    ./run.sh $num1 $num2
done

rm -f input/*
rm -f output/*