import matplotlib.pyplot as plt
import matplotlib.animation as animation

def fibonacci_animation(n):
    """
    Animates the step-by-step computation of the first n Fibonacci numbers.
    
    Parameters:
    n (int): Number of Fibonacci numbers to compute (n >= 2)
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    
    # Initialize Fibonacci sequence
    fib = [0, 1]
    
    # Precompute all Fibonacci numbers up to n
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, n-1)
    ax.set_ylim(0, max(fib) * 1.1)
    ax.set_xlabel('Step (Index)')
    ax.set_ylabel('Fibonacci Value')
    ax.set_title('Fibonacci Sequence Animation')
    ax.grid(True)
    
    # Initialize empty plot elements
    bars_list = []
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    def animate(frame):
        # Clear previous bars
        for bar in bars_list:
            bar.remove()
        bars_list.clear()
        
        # Create new bars for current frame
        x = list(range(frame + 1))
        y = fib[:frame + 1]
        for i, height in enumerate(y):
            bar = ax.bar(i, height, color='skyblue', edgecolor='black')[0]
            bars_list.append(bar)
        
        # Update step text
        if frame < 2:
            text.set_text(f'Step {frame}: F({frame}) = {fib[frame]} (Base case)')
        else:
            text.set_text(f'Step {frame}: F({frame}) = F({frame-1}) + F({frame-2}) = {fib[frame-1]} + {fib[frame-2]} = {fib[frame]}')
        return bars_list + [text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        animate, 
        frames=n, 
        interval=1500,  # 1.5 seconds per step
        blit=False, 
        repeat=False
    )
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    fibonacci_animation(8)