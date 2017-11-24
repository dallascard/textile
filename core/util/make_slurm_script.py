import os
from optparse import OptionParser


def main():
    usage = "%prog input.sh output.sh [additional modules]"
    parser = OptionParser(usage=usage)
    parser.add_option('-N', dest='nodes', default=1,
                      help='Number of nodes required: default=%default')
    parser.add_option('-n', dest='tasks', default=1,
                      help='Total number of tasks: default=%default')
    parser.add_option('-t', dest='hours', default=8,
                      help='Estimated number of hours required: default=%default')
    parser.add_option('--dev', action="store_true", dest="dev", default=False,
                      help='Use development queue: default=%default')
    parser.add_option('-m', dest='module', default='pytorch',
                      help='Base module: default=%default')

    (options, args) = parser.parse_args()
    input_file = args[0]
    output_file = args[1]
    if len(args) > 2:
        additional_modules = args[2:]
    else:
        additional_modules = []

    nodes = int(options.nodes)
    tasks = int(options.tasks)
    hours = int(options.hours)
    dev = options.dev
    module = options.module

    basename = os.path.basename(input_file)
    name = os.path.splitext(basename)[0]

    with open(input_file, 'r') as f:
        input_text = f.read()

    script = make_script(name, input_text, nodes, tasks, hours, dev, module, additional_modules)

    with open(output_file, 'w') as f:
        f.write(script)


def make_script(name, input_text, nodes, tasks, hours, dev, module, addtional_modules):

    script = """#!/bin/bash
#-----------------------------------------------------------------
# Example SLURM job script to run serial applications on TACC's
# Stampede system.
#
# This script requests one core (out of 16) on one node. The job
# will have access to all the memory in the node.  Note that this
# job will be charged as if all 16 cores were requested.
#-----------------------------------------------------------------
    
#SBATCH --mail-user=dallas.slurm@yahoo.ca
#SBATCH --mail-type=all
"""
    script += "#SBATCH -J " + name + "\n"
    script += "#SBATCH -o " + name + ".%j.out\n"
    script += "#SBATCH -n " + str(tasks) + "\n"
    if dev:
        script += "#SBATCH -p development\n"
    else:
        script += "#SBATCH -p normal\n"
    script += "#SBATCH -N " + str(nodes) + "\n"
    script += "#SBATCH -t " + str(hours) + ":30:00\n"

    script += "\ndate\n"
    script += "source activate " + str(module) + "\n"
    for mod in addtional_modules:
        script += "source activate " + str(mod) + "\n"
    for line in input_text:
        script += line
    script += "\ndate\n"

    return script


if __name__ == '__main__':
    main()

