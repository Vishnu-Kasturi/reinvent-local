# Place your receptor and reference ligand here.
#
# Setup:
#   1. Copy receptor.pdb into this directory
#   2. Copy co-crystallized ligand as ref_ligand.sdf (optional, for autobox)
#   3. Run:
#      python Preprocess/scripts/prepare_receptor_for_gnina.py \
#          --input docking/receptor.pdb \
#          --output docking/receptor.pdbqt \
#          --ref_ligand docking/ref_ligand.sdf \
#          --write_grid docking/grid.json
#   4. Update center/size or autobox_ligand paths in your TOML config
#
# Files (not committed — add to .gitignore if large):
#   receptor.pdb
#   receptor.pdbqt
#   ref_ligand.sdf
#   grid.json
#   gnina_cache/
