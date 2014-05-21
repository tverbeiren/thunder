import AssemblyKeys._

assemblySettings

name := "Thunder"

version := "0.1.0"

scalaVersion := "2.10.3"

ivyXML := <dependency org="org.eclipse.jetty.orbit" name="javax.servlet" rev="2.5.0.v201103041518">
<artifact name="javax.servlet" type="orbit" ext="jar"/>
</dependency>

libraryDependencies += "org.apache.spark" %% "spark-core" % "0.9.0-incubating"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "0.9.0-incubating"

libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "0.9.0-incubating"

libraryDependencies += "org.apache.spark" % "spark-streaming-kafka_2.10" % "0.9.0-incubating"

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test"

libraryDependencies += "io.spray" %% "spray-json" % "1.2.5"

libraryDependencies += "org.jblas" % "jblas" % "1.2.3"

resolvers += "spray" at "http://repo.spray.io/"

resolvers ++= Seq(
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Spray Repository" at "http://repo.spray.cc/")


mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
{
  case PathList("javax", "servlet", xs @ _*) => MergeStrategy.last
  case PathList("org", "apache", xs @ _*) => MergeStrategy.last
  case PathList("com", "esotericsoftware", xs @ _*) => MergeStrategy.last
  case PathList("org", "slf4j", xs @ _*) => MergeStrategy.last
  case "about.html" => MergeStrategy.rename
  case x => old(x)
}
}


