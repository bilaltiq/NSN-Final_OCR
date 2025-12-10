from accuracy import test_accuracy

def main():
    Epsilon = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1]
    dataset = "./Dataset/GT-pairs"
    output = "./Dataset/Outputs"
    
    # These to metrics are specifically for paged gradient descent
    step_size = 0.025
    iters = 10

    # Run tests
    for epsilon in Epsilon: 
        test_accuracy(dataset, output, epsilon, step_size, iters)

    print("Experiment Complete for PP_OCRv5 English\n")

if __name__ == "__main__":
    main()