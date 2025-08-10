import os

def print_environment_variables():
    """Prints all environment variables."""
    print("Environment Variables:")
    for env in list(os.environ):
        print(f'{env}={os.environ[env]}')



if __name__ == "__main__":
    print_environment_variables()