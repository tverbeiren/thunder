package thunder.streaming

import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import thunder.util.Load

object KafkaExample {

  def main(args: Array[String]) {

    val master = args(0)
    val batchTime = args(1).toLong

    val conf = new SparkConf().setMaster(master).setAppName("KafkaExample")
    val ssc = new StreamingContext(conf, Seconds(batchTime))

    val data = Load.loadStreamingDataFromKafka(ssc)

    data.print()

    ssc.start()

  }

}
