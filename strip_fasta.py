import sys

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 strip_fasta.py {{FASTA_FILE}} {{OUTPUT_FILE}}")
        exit()

    proteins = []

    with open(sys.argv[1], 'r') as fasta:
        for line in fasta:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
                protein_name = header.split()[0]
                proteins.append(protein_name)

    with open(sys.argv[2], 'w') as protein_file:
        for p in proteins:
            protein_file.write(f"{p}\n")
    pass