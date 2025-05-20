import json
from pathlib import Path
from simple_client import SimpleClient
import time

def run_similarity_test():
    # Get the script's directory
    script_dir = Path(__file__).parent.absolute()
    config_path = script_dir / "config.json"
    
    # Create a new task directory
    client = SimpleClient(config_path)
    
    # Process the test file
    input_file = client.base_path / "inputs" / "test_similarity.json"
    
    print("\nStarting similarity test...")
    print("This test will process similar prompts and demonstrate the similarity checking functionality.")
    print("You should see cooldown periods being applied when similar responses are detected.\n")
    
    try:
        # Process with OpenAI
        if client.openai_client:
            print("\nProcessing with OpenAI...")
            start_time = time.time()
            result = client.process_batch(input_file, service="openai")
            end_time = time.time()
            
            print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
            print(f"Processed {len(result)} responses")
            
            # Print cache stats
            cache_stats = client.get_cache_stats()
            print("\nCache Statistics:")
            print(json.dumps(cache_stats, indent=2))
            
            # Print summary
            print("\nTest Summary:")
            print("1. The test file contains 9 prompts grouped into 3 similar sets")
            print("2. Each set contains 3 similar prompts that should trigger similarity detection")
            print("3. You should have seen cooldown periods being applied when similar responses were detected")
            print("4. The cache should show hits for similar responses")
            
    except Exception as e:
        print(f"Error during test: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    run_similarity_test() 