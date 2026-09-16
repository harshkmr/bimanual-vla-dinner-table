# TICKET-04 primary path — SmolVLA finetune via LeRobot (cloud GPU).
# TODO_TICKET=TICKET-04: fill HF_USER, steps, batch per GPU (16/4090, 32-64/A40/A100).
# lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=${HF_USER}/so101-bimanual-table --steps 20000 --batch_size 32
echo "TICKET-04: configure + run on cloud GPU; push HF_USER/so101-bimanual-table-smolvla"
