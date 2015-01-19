"""
Wrapper for the Spark EC2 launch script that additionally
installs Thunder and its dependencies, and optionally
loads an example data set
"""

from boto import ec2
import sys
import os
import random
import subprocess
from sys import stderr
from optparse import OptionParser
from spark_ec2 import ssh, launch_cluster, get_existing_cluster, wait_for_cluster, deploy_files, setup_spark_cluster, \
    get_spark_ami, ssh_command, ssh_read, ssh_write, get_or_make_group


def get_s3_keys():
    """ Get user S3 keys from environmental variables"""
    if os.getenv('S3_AWS_ACCESS_KEY_ID') is not None:
        s3_access_key = os.getenv("S3_AWS_ACCESS_KEY_ID")
    else:
        s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    if os.getenv('S3_AWS_SECRET_ACCESS_KEY') is not None:
        s3_secret_key = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
    else:
        s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    return s3_access_key, s3_secret_key


def install_thunder(master, opts):
    """ Install Thunder and dependencies on a Spark EC2 cluster"""
    print "Installing Thunder on the cluster..."
    # download and build thunder
    ssh(master, opts, "rm -rf thunder && git clone https://github.com/freeman-lab/thunder.git")
    ssh(master, opts, "chmod u+x thunder/python/bin/build")
    ssh(master, opts, "thunder/python/bin/build")
    # copy local data examples to all workers
    ssh(master, opts, "yum install -y pssh")
    ssh(master, opts, "pssh -h /root/spark-ec2/slaves mkdir -p /root/thunder/python/thunder/utils/data/")
    ssh(master, opts, "~/spark-ec2/copy-dir /root/thunder/python/thunder/utils/data/")
    # install pip
    ssh(master, opts, "wget http://pypi.python.org/packages/source/p/pip/pip-1.1.tar.gz"
                      "#md5=62a9f08dd5dc69d76734568a6c040508")
    ssh(master, opts, "tar -xvf pip*.gz")
    ssh(master, opts, "cd pip* && sudo python setup.py install")
    # install libraries
    ssh(master, opts, "source ~/.bash_profile && pip install mpld3 && pip install seaborn "
                      "&& pip install jinja2 && pip install -U scikit-learn")
    # install ipython 1.1
    #ssh(master, opts, "pip uninstall -y ipython")
    ssh(master, opts, "git clone https://github.com/ipython/ipython.git")
    ssh(master, opts, "cd ipython && git checkout tags/rel-1.1.0")
    ssh(master, opts, "cd ipython && sudo python setup.py install")
    # set environmental variables
    ssh(master, opts, "echo 'export SPARK_HOME=/root/spark' >> /root/.bash_profile")
    ssh(master, opts, "echo 'export PYTHONPATH=/root/thunder/python' >> /root/.bash_profile")
    ssh(master, opts, "echo 'export IPYTHON=1' >> /root/.bash_profile")
    ssh(master, opts, "echo 'export PATH=/root/thunder/python/bin:$PATH' >> /root/.bash_profile")
    # customize spark configuration parameters
    ssh(master, opts, "echo 'spark.akka.frameSize=10000' >> /root/spark/conf/spark-defaults.conf")
    ssh(master, opts, "echo 'spark.kryoserializer.buffer.max.mb=1024' >> /root/spark/conf/spark-defaults.conf")
    ssh(master, opts, "echo 'export SPARK_DRIVER_MEMORY=20g' >> /root/spark/conf/spark-env.sh")
    # add AWS credentials to core-site.xml
    configstring = "<property><name>fs.s3n.awsAccessKeyId</name><value>ACCESS</value></property><property>" \
                   "<name>fs.s3n.awsSecretAccessKey</name><value>SECRET</value></property>"
    access, secret = get_s3_keys()
    filled = configstring.replace('ACCESS', access).replace('SECRET', secret)
    ssh(master, opts, "sed -i'f' 's,.*</configuration>.*,"+filled+"&,' /root/ephemeral-hdfs/conf/core-site.xml")
    # add AWS credentials to ~/.boto
    credentialstring = "[Credentials]\naws_access_key_id = ACCESS\naws_secret_access_key = SECRET\n"
    credentialsfilled = credentialstring.replace('ACCESS', access).replace('SECRET', secret)
    ssh(master, opts, "printf '"+credentialsfilled+"' > /root/.boto")
    ssh(master, opts, "pscp.pssh -h /root/spark-ec2/slaves /root/.boto /root/.boto")
    # configure requester pays
    ssh(master, opts, "touch /root/spark/conf/jets3t.properties")
    ssh(master, opts, "echo 'httpclient.requester-pays-buckets-enabled = true' >> /root/spark/conf/jets3t.properties")
    ssh(master, opts, "~/spark-ec2/copy-dir /root/spark/conf")

    print "\n\n"
    print "-------------------------------"
    print "Thunder successfully installed!"
    print "-------------------------------"
    print "\n"


def setup_cluster(conn, master_nodes, slave_nodes, opts, deploy_ssh_key):
    """Modified version of the setup_cluster function (borrowed from spark-ec.py)
    in order to manually set the folder with the deploy code
    """
    master = master_nodes[0].public_dns_name
    if deploy_ssh_key:
        print "Generating cluster's SSH key on master..."
        key_setup = """
      [ -f ~/.ssh/id_rsa ] ||
        (ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa &&
         cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys)
        """
        ssh(master, opts, key_setup)
        dot_ssh_tar = ssh_read(master, opts, ['tar', 'c', '.ssh'])
        print "Transferring cluster's SSH key to slaves..."
        for slave in slave_nodes:
            print slave.public_dns_name
            ssh_write(slave.public_dns_name, opts, ['tar', 'x'], dot_ssh_tar)

    modules = ['spark', 'shark', 'ephemeral-hdfs', 'persistent-hdfs',
               'mapreduce', 'spark-standalone', 'tachyon']

    if opts.hadoop_major_version == "1":
        modules = filter(lambda x: x != "mapreduce", modules)

    if opts.ganglia:
        modules.append('ganglia')

    ssh(master, opts, "rm -rf spark-ec2 && git clone https://github.com/mesos/spark-ec2.git -b v3")

    print "Deploying files to master..."
    deploy_folder = os.path.join(os.environ['SPARK_HOME'], "ec2", "deploy.generic")
    deploy_files(conn, deploy_folder, opts, master_nodes, slave_nodes, modules)

    print "Running setup on master..."
    setup_spark_cluster(master, opts)
    print "Done!"


if __name__ == "__main__":
    parser = OptionParser(usage="thunder-ec2 [options] <action> <clustername>",  add_help_option=False)
    parser.add_option("-h", "--help", action="help", help="Show this help message and exit")
    parser.add_option("-k", "--key-pair", help="Key pair to use on instances")
    parser.add_option("-s", "--slaves", type="int", default=1, help="Number of slaves to launch (default: 1)")
    parser.add_option("-i", "--identity-file", help="SSH private key file to use for logging into instances")
    parser.add_option("-r", "--region", default="us-east-1", help="EC2 region zone to launch instances "
                                                                  "in (default: us-east-1)")
    parser.add_option("-t", "--instance-type", default="m3.2xlarge",
                      help="Type of instance to launch (default: m3.2xlarge)." +
                           " WARNING: must be 64-bit; small instances won't work")
    parser.add_option("-u", "--user", default="root", help="User name for cluster (default: root)")
    parser.add_option("-w", "--wait", type="int", default=160,
                      help="Seconds to wait for nodes to start (default: 160)")
    parser.add_option("-z", "--zone", default="", help="Availability zone to launch instances in, or 'all' to spread "
                                                       "slaves across multiple (an additional $0.01/Gb for "
                                                       "bandwidth between zones applies)")
    parser.add_option("--spot-price", metavar="PRICE", type="float",
                      help="If specified, launch slaves as spot instances with the given " +
                           "maximum price (in dollars)")
    parser.add_option("--resume", default=False, action="store_true",
                      help="Resume installation on a previously launched cluster (for debugging)")

    (opts, args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(1)
    (action, cluster_name) = args

    # Launch a cluster, setting several options to defaults
    # (use spark-ec2.py included with Spark for more control)
    if action == "launch":
        try:
            conn = ec2.connect_to_region(opts.region)
        except Exception as e:
            print >> stderr, (e)
            sys.exit(1)

        if opts.zone == "":
            opts.zone = random.choice(conn.get_all_zones()).name

        opts.ami = get_spark_ami(opts)  # "ami-3ecd0c56"
        opts.ebs_vol_size = 0
        opts.master_instance_type = ""
        opts.hadoop_major_version = "1"
        opts.ganglia = True
        opts.spark_version = "1.1.0"
        opts.swap = 1024
        opts.worker_instances = 1
        opts.master_opts = ""
        opts.user_data = ""

        if opts.resume:
            (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name)
        else:
            (master_nodes, slave_nodes) = launch_cluster(conn, opts, cluster_name)

        wait_for_cluster(conn, opts.wait, master_nodes, slave_nodes)
        setup_cluster(conn, master_nodes, slave_nodes, opts, True)
        master = master_nodes[0].public_dns_name
        install_thunder(master, opts)
        print "\n\n"
        print "-------------------------------"
        print "Cluster successfully launched!"
        print "Go to http://%s:8080 to see the web UI for your cluster" % master
        print "-------------------------------"
        print "\n"

    if action != "launch":
        conn = ec2.connect_to_region(opts.region)
        (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name)
        master = master_nodes[0].public_dns_name

        # Login to the cluster
        if action == "login":
            print "Logging into master " + master + "..."
            proxy_opt = []
            subprocess.check_call(ssh_command(opts) + proxy_opt + ['-t', '-t', "%s@%s" % (opts.user, master)])

        # Install thunder on the cluster
        elif action == "install":
            install_thunder(master, opts)

        # Destroy the cluster
        elif action == "destroy":
            response = raw_input("Are you sure you want to destroy the cluster " + cluster_name +
                                 "?\nALL DATA ON ALL NODES WILL BE LOST!!\n" +
                                 "Destroy cluster " + cluster_name + " (y/N): ")
            if response == "y":
                (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name, die_on_error=False)
            print "Terminating master..."
            for inst in master_nodes:
                inst.terminate()
            print "Terminating slaves..."
            for inst in slave_nodes:
                inst.terminate()

        else:
            raise NotImplementedError("action: " + action + "not recognized")
