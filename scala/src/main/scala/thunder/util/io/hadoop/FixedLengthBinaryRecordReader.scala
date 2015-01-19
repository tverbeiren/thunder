package thunder.util.io.hadoop

import java.io.IOException
import scala.util.control.Breaks._
import org.apache.hadoop.fs.FSDataInputStream
import org.apache.hadoop.io.{BytesWritable, LongWritable}
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.hadoop.mapreduce.InputSplit
import org.apache.hadoop.mapreduce.lib.input.FileSplit
import org.apache.hadoop.mapreduce.RecordReader
import org.apache.hadoop.mapreduce.TaskAttemptContext

/**
 *
 * FixedLengthBinaryRecordReader is returned by FixedLengthBinaryInputFormat.
 * It uses the record length set in FixedLengthBinaryInputFormat to
 * read one record at a time from the given InputSplit.
 *
 * Each call to nextKeyValue() updates the LongWritable KEY and BytesWritable VALUE.
 *
 * KEY = record index (Long)
 * VALUE = the record itself (BytesWritable)
 *
 */
class FixedLengthBinaryRecordReader extends RecordReader[LongWritable, BytesWritable] {

  override def close() {
    if (fileInputStream != null) {
      fileInputStream.close()
    }
  }

  override def getCurrentKey: LongWritable = {
    recordKey
  }

  override def getCurrentValue: BytesWritable = {
    recordValue
  }

  override def getProgress: Float = {
    splitStart match {
      case x if x == splitEnd => 0.0.toFloat
      case _ => Math.min(((currentPosition - splitStart) / (splitEnd - splitStart)).toFloat, 1.0).toFloat
    }
  }

  override def initialize(inputSplit: InputSplit, context: TaskAttemptContext) {

    // the file input
    val fileSplit = inputSplit.asInstanceOf[FileSplit]

    // the byte position this fileSplit starts at
    splitStart = fileSplit.getStart

    // splitEnd byte marker that the fileSplit ends at
    splitEnd = splitStart + fileSplit.getLength

    // the actual file we will be reading from
    val file = fileSplit.getPath

    // job configuration
    val job = context.getConfiguration

    // check compression
    val codec = new CompressionCodecFactory(job).getCodec(file)
    if (codec != null) {
      throw new IOException("FixedLengthRecordReader does not support reading compressed files")
    }

    // get the record length
    recordLength = FixedLengthBinaryInputFormat.getRecordLength(context)

    // get the filesystem
    val fs = file.getFileSystem(job)

    // open the File
    fileInputStream = fs.open(file)

    // seek to the splitStart position
    fileInputStream.seek(splitStart)

    // set our current position
    currentPosition = splitStart

  }

  override def nextKeyValue(): Boolean = {

    if (recordKey == null) {
      recordKey = new LongWritable()
    }

    // the key is a linear index of the record, given by the
    // position the record starts divided by the record length
    recordKey.set(currentPosition / recordLength)

    // the recordValue to place the bytes into
    if (recordValue == null) {
      recordValue = new BytesWritable(new Array[Byte](recordLength))
    }

    // read a record if the currentPosition is less than the split end
    if (currentPosition < splitEnd) {

      // setup a buffer to store the record
      val buffer = recordValue.getBytes
      fileInputStream.readFully(buffer)

      // update our current position
      currentPosition += recordLength

      // return true
      return true
    }

    false
  }

  var splitStart: Long = 0L
  var splitEnd: Long = 0L
  var currentPosition: Long = 0L
  var recordLength: Int = 0
  var fileInputStream: FSDataInputStream = null
  var recordKey: LongWritable = null
  var recordValue: BytesWritable = null

}