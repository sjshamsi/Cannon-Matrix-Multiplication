#!/bin/bash
# Usage:
#   ./pmmult.sh <nodes> <tasks_per_node> <cpus_per_task> <omp_threads> <openblas_threads> <matrix_order>
#
# Example:
#   ./pmmult.sh 2 4 8 4 1 1024

set -euo pipefail

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <nodes> <tasks_per_node> <cpus_per_task> <omp_threads> <openblas_threads> <matrix_order>"
    exit 1
fi

# Various settings
NODES="$1"
TASKS_PER_NODE="$2"
CPUS_PER_TASK="$3"
OMP_NUM_THREADS="$4"
OPENBLAS_NUM_THREADS="$5"
ORDER="$6"

PROJ_NAME="PMatMult"
BUILD="REL"
OPENMPI_MOD="openmpi"
BLAS_MOD="openblas"
REPO_ROOT="mpi-cannon-matmul"   # path of this repo under $HOME
ACCOUNT="${ACCOUNT:?set ACCOUNT to your Slurm account}"
PROGRAM_NAME="${PROGRAM_NAME:-ParallelCMatrix.x}"   # e.g. PROGRAM_NAME=CanonM_3Buff.x

# Derived stuff
OUTDIR_PARENT="${HOME}/Jobs/${PROJ_NAME}"
COMBINATION="n${NODES}_tpn${TASKS_PER_NODE}_cpt${CPUS_PER_TASK}_ompt${OMP_NUM_THREADS}_obt${OPENBLAS_NUM_THREADS}_o${ORDER}"

OUTDIR="${OUTDIR_PARENT}/o${ORDER}/${COMBINATION}"
WORK_DIR=$(mktemp -d "${SCRATCH}/${PROJ_NAME}_${COMBINATION}_XXXXXX")

TIME=$(date +"%d-%m_%H-%M-%S")
SRC_DIR="${WORK_DIR}/${REPO_ROOT}"
BUILD_DIR="${SRC_DIR}/build/${BUILD}"
PROGRAM="${BUILD_DIR}/bin/${PROGRAM_NAME}"

mkdir -p "${OUTDIR}"
mkdir -p "${BUILD_DIR}"

cp -rf "${HOME}/${REPO_ROOT}/src" "${SRC_DIR}/"
cp -rf "${HOME}/${REPO_ROOT}/inc" "${SRC_DIR}/"
cp -rf "${HOME}/${REPO_ROOT}/CMakeLists.txt" "${SRC_DIR}/"
cp -rf "${HOME}/${REPO_ROOT}/CMakePresets.json" "${SRC_DIR}/"
cp -rf "${HOME}/${REPO_ROOT}/external" "${SRC_DIR}/"


JOB_SCRIPT=$(mktemp /tmp/pmmult_job_XXXXXX.sh)
cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name="${PROJ_NAME}_${COMBINATION}"
#SBATCH --time=00:30:00
#SBATCH --nodes="${NODES}"
#SBATCH --ntasks-per-node="${TASKS_PER_NODE}"
#SBATCH --cpus-per-task="${CPUS_PER_TASK}"
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --output=${OUTDIR}/${TIME}_%j.out
#SBATCH --error=${OUTDIR}/${TIME}_%j.err
#SBATCH --account=${ACCOUNT}

set -euo pipefail

OUTFILE="${OUTDIR}/${TIME}_\${SLURM_JOB_ID}.out"
TIMEFILE="${OUTDIR}/${TIME}_\${SLURM_JOB_ID}.times"


echo " ==== Loading modules... ===== "
module load cmake
module load ninja
module load "${BLAS_MOD}"
module load "${OPENMPI_MOD}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}"


echo " ==== Building Project... ===== "
cmake --preset "${BUILD}" -S "${SRC_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}"


echo " ==== Outputs ===== "
{
  echo "#Date_Time:               \$(date)"
  echo "#Hostname:                \$(hostname)"
  echo "#"
  echo "#SLURM_JOB_NAME:          \${SLURM_JOB_NAME}"
  echo "#SLURM_JOB_ID:            \${SLURM_JOB_ID}"
  echo "#SLURM_JOB_NUM_NODES:     \${SLURM_JOB_NUM_NODES}"
  echo "#SLURM_TASKS_PER_NODE:    \${SLURM_TASKS_PER_NODE}"
  echo "#SLURM_CPUS_PER_TASK:     \${SLURM_CPUS_PER_TASK}"
  echo "#SLURM_NTASKS:            \${SLURM_NTASKS}"
  echo "#SLURM_JOB_CPUS_PER_NODE: \${SLURM_JOB_CPUS_PER_NODE}"
  echo "#SLURM_NODELIST:          \${SLURM_NODELIST}"
  echo "#ENV_OMP_NUM_Threads:     \${OMP_NUM_THREADS}"
  echo "#ENV_OB_NUM_Threads:      \${OPENBLAS_NUM_THREADS}"
  echo "#"
  echo "#Requested_Nodes:         ${NODES}"
  echo "#Requested_TPN:           ${TASKS_PER_NODE}"
  echo "#Requested_CPT:           ${CPUS_PER_TASK}"
  echo "#Exported_OMP_Threads:    ${OMP_NUM_THREADS}"
  echo "#Exported_OB_Threads:     ${OPENBLAS_NUM_THREADS}"
  echo "#Matrix_Order:            ${ORDER}"
} >> "\${TIMEFILE}"

echo "" >> "\${TIMEFILE}"
cat "\${TIMEFILE}" >> "\${OUTFILE}"


echo " ==== Running Program... ===== "

srun --cpu-bind=cores "${PROGRAM}" -N ${ORDER} >> "\${TIMEFILE}"
echo "Program output redirected to: \${OUTFILE} and \${TIMEFILE}"

echo "==== Cleanup... ==== "
rm -rf "${WORK_DIR}"
echo "==== Cleanup Done! ==== "
EOF

chmod +x "${JOB_SCRIPT}"
JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $4}')

cp "${JOB_SCRIPT}" "${OUTDIR}/${TIME}_${JOB_ID}.slurm"

echo "Submitted job:"
echo "  Nodes=${NODES}, TPN=${TASKS_PER_NODE}, CPT=${CPUS_PER_TASK}, OMP_Threads="${OMP_NUM_THREADS}" OB_Threads="${OPENBLAS_NUM_THREADS}", MatrixDim=${ORDER}x${ORDER}"
echo "with Job ID ${JOB_ID} using ${JOB_SCRIPT}"