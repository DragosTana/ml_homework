import kernel as ker

def main():
    
    gauss = ker.kernel("gaussian")
    value = gauss.compute_kernel(1)
    print(value)
    
    
    
if __name__ == "__main__":
    main()