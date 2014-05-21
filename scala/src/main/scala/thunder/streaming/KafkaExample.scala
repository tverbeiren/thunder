package thunder.streaming

import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import thunder.util.Load

object KafkaExample {

  def main(args: Array[String]) {

    val master = args(0)
    val batchTime = args(1).toLong

    val conf = new SparkConf().setMaster(master).setAppName("KafkaExample")

    if (!master.contains("local")) {
      conf.setSparkHome(System.getenv("SPARK_HOME"))
        .setJars(List("target/scala-2.10/Thunder-assembly-0.1.0.jar"))
        .set("spark.executor.memory", "100G")
        .set("spark.default.parallelism", "100")
    }

    val ssc = new StreamingContext(conf, Seconds(batchTime))

    val data = Load.loadStreamingDataFromKafka(ssc).repartition(100).foreachRDD(rdd => print("\n \n %s \n \n".format(rdd.count())))

    ssc.start()

  }

}
