from rich.console import Console
from loguru import logger

console = Console()

def main():
    console.print("[bold cyan]EPSILON ONLINE[/bold cyan]")
    logger.info("System boot sequence completed...")

if __name__ == "main":
    main()