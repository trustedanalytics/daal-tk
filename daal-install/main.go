package main

import (
	"archive/zip"
	"bytes"
	"fmt"
	//"io/ioutil"
	"io"
	"os"
	"os/exec"
	"os/user"
	"strconv"

	log "github.com/Sirupsen/logrus"
)

const (
	FILEMODE        = os.FileMode(int(0755))
	DAAL_VERSION    = "2016.2.181"
	DAALTK_VERSION  = "0.7.3.384"
	SPARKTK_VERSION = "0.7.3.dev2614"
)

var (
	HOME               = ""
	MANDATORY_ENV      = []string{"SPARK_HOME"}
	SPARKTK_CORE_PKG   = fmt.Sprintf("sparktk-core-%s.zip", SPARKTK_VERSION)
	SPARKTK_PYTHON_PKG = fmt.Sprintf("sparktk-%s.tar.gz", SPARKTK_VERSION)
	DAALTK_CORE_PKG    = fmt.Sprintf("daaltk-core-%s.zip", DAALTK_VERSION)
	DAALTK_PYTHON_PKG  = fmt.Sprintf("daaltk-%s.tar.gz", DAALTK_VERSION)
	DAAL_PKG           = fmt.Sprintf("daal-%s.zip", DAAL_VERSION)
)

func main() {
	checkEnv()
	exports := []string{}
	who, err := user.Current()
	if err != nil {
		log.Fatal(err)
	}
	//we have a root user
	if who.Uid == strconv.FormatInt(0, 10) {
		HOME = "/usr/local"
	} else {
		HOME = who.HomeDir
	}

	//unzip sparktk-core
	sparktkCore, err := Asset(SPARKTK_CORE_PKG)
	if err != nil {
		fmt.Println("Asset not found ", SPARKTK_CORE_PKG)
	}
	fmt.Printf("Extract Spark-tk to %s \n", HOME)
	unzip(HOME, sparktkCore)
	exports = append(exports, exportEnv("SPARKTK_HOME", fmt.Sprintf("%s/sparktk-core-%s", HOME, SPARKTK_VERSION)))

	//install sparktk python package
	sparktkPython, err := Asset(SPARKTK_PYTHON_PKG)
	if err != nil {
		fmt.Println("Asset not found ", SPARKTK_PYTHON_PKG)
	}
	fmt.Println("Install Spark-tk python package ")
	saveFile(SPARKTK_PYTHON_PKG, sparktkPython)

	pipInstall(SPARKTK_PYTHON_PKG)

	//unzip daaltk-core
	daaltkCore, err := Asset(DAALTK_CORE_PKG)
	if err != nil {
		fmt.Println("Asset not found ", DAALTK_CORE_PKG)
	}
	fmt.Printf("Extract Daal-tk to %s \n", HOME)
	unzip(HOME, daaltkCore)
	exports = append(exports, exportEnv("DAALTK_HOME", fmt.Sprintf("%s/daaltk-core-%s", HOME, DAALTK_VERSION)))

	//install daaltk python package
	daaltkPython, err := Asset(DAALTK_PYTHON_PKG)
	if err != nil {
		fmt.Println("Asset not found ", DAALTK_PYTHON_PKG)
	}
	fmt.Println("Install Daal-tk python package ")
	saveFile(DAALTK_PYTHON_PKG, daaltkPython)
	pipInstall(DAALTK_PYTHON_PKG)
	/*
		//install daal libraries
	*/
	daal, err := Asset(DAAL_PKG)
	if err != nil {
		fmt.Println("Asset not found ", DAAL_PKG)
	}
	fmt.Printf("Extract Daal to %s \n", HOME)
	unzip(HOME, daal)
	exports = append(exports, exportEnv("LD_LIBRARY_PATH", fmt.Sprintf("%s/daal-%s:$LD_LIBRARY_PATH", HOME, DAAL_VERSION)))

	fmt.Println("The following exports must be set in your environment.")
	for _, export := range exports {
		fmt.Println(export)
	}
}

func checkEnv() {
	for _, env := range MANDATORY_ENV {
		if _, ok := os.LookupEnv(env); !ok {
			log.Warning(fmt.Sprintf("The %s env configuration is mandatory and must be set for you to run Spark-tk,Daal-tk.", env))
		}
	}

}
func pipInstall(name string) {
	pipInstall := exec.Command("pip2.7", "install", "-U", getPath(name))
	out, err := pipInstall.CombinedOutput()
	if err != nil {
		log.Error(err)
	}
	fmt.Printf("%s\n", out)
}
func saveFile(name string, fileBytes []byte) {
	newFile, err := os.Create(getPath(name))
	if err != nil {
		log.Fatal(err)
	}

	newFile.Write(fileBytes)
	newFile.Close()
}
func unzip(baseDir string, zipBytes []byte) {
	reader := bytes.NewReader(zipBytes)

	zipReader, err := zip.NewReader(reader, int64(len(zipBytes)))
	if err != nil {
		fmt.Println("can't open zip file")
	}
	for _, zFile := range zipReader.File {
		fmt.Println(zFile.Name)
		filePath := fmt.Sprintf("%s/%s", baseDir, zFile.Name)
		fileInfo := zFile.FileInfo()

		if fileInfo.IsDir() {
			os.MkdirAll(filePath, FILEMODE)
		} else {
			fp, err := zFile.Open()
			if err != nil {
				log.Fatal(err)
			}
			defer fp.Close()

			f, err := os.Create(filePath)
			if err != nil {
				log.Fatal(err)
			}
			defer f.Close()

			io.Copy(f, fp)
			f.Close()
		}
	}
}

func exportEnv(name, value string) string {
	err := os.Setenv(name, value)
	if err != nil {
		log.Fatal(err)
	}
	return fmt.Sprintf("export %s=\"%s\" ", name, value)
}
func getPath(fileName string) string {
	fmt.Printf("%s/%s", HOME, fileName)
	return fmt.Sprintf("%s/%s", HOME, fileName)
}
