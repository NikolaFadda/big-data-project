# big-data-project
Nicola Fadda 65187, 2020/2021

GUIDA ALL'INSTALLAZIONE

 - Eseguire la pull del progetto
 - Scaricare il database con gli articoli al seguente link nella cartella del progetto: 
 ```console 
 https://drive.google.com/file/d/148fJCujxEKYetl8twrV31VZgcGbhKLqj/view?usp=sharing 
 ```
 - Scaricare il pacchetto Terraform al seguente link nella cartella del progetto, ed estrarlo: 
```console 
https://drive.google.com/file/d/1llRpS8RPhW7M67uTGdqvLMJ61S68Kc3G/view?usp=sharing 
```
 - Eliminare l'archivio terraform.zip dopo che l'estrazione è avvenuta con successo
 - Creare un nuovo utente IAM su Amazon Web Services: 
    - Accedere alla sezione Servizi
    - All'interno dell'elenco "Servizi, sicurezza, identità e conformità" cliccare su IAM
    - Cliccare su "Utenti" e poi su "Aggiungi Utente"
    - Inserire uno Username a scelta
    - Impostare l'access type su "Programmatic Access" e proseguire (clic su "Next")
    - Selezionare "Attach Existing Policies Directly" e spuntare "Administrator Access"
    - Continuare a cliccare "Next" fichè non è possibile creare l'utente
    - Salvare le nuove credenziali in formato csv
    
- Installare "awscli" da terminale con il comando:
    
    ```console
    user@user:~$ sudo apt install awscli
    ```
 
- Dopo aver concluso l'installazione di "aws cli", configurar le credenziali in locale con:

    ```console
    user@user:~$ aws configure
    ```
    
    - Avere cura di inserire per primo l'access_key_id e poi la secret_access_key_id presenti nel file csv appena scaricato, "region name" impostato a "eu-west-3" e "output format" a "json"
    
- Creare una nuova coppia di chiavi su AWS in formato ".pem", da salvare in locale:
    
    - Accedere alla sezione Servizi
    - Selezionare "EC2"
    - Selezionare "Key Pair" in "Network & Security"
    - Selezionare "Creare Key Pair"
    - Assegnare un nome a scelta e selezionare il formato file PEM
    - Cliccare su "Crea" e salvare la chiave PEM nella cartella del progetto
    
    Cambiare i permessi della chiave.pem, accedendo alla cartella in cui è salvata (quindi quella del progetto):
    ```console
    user@user:~$ chmod 400 my-key-pair.pem
    ```
- Entrare nella cartella "terraform"
    - Modificare il file "terraform.tfvars": al campo "access_key_name" inserire il nome della chiave SENZA l'estensione ".pem"
    - Modificare il file "terraform.tfvars": al campo "access_key_path" inserire l'indirizzo locale della chiave, compreso nome della stessa ed estensione ".pem"
    - Modificare a piacimento il file "config.tf": il setup attuale creerà 1 macchina master tipo t2.medium e 6 macchine slave tipo t2.micro. Per rimuovere slave basta eliminare i blocchi:
    ```console
    resource "aws_instance" "slaveN"  {
          . . .
          #N è il numero che indica l'N-esimo slave
    }
    ```
- Eseguire i seguenti comandi all'interno della cartella "terraform":
    
    - Inizializzare terraform:
    ```console
    user@user:~$ ./terraform init
    ```
    
    - Verificare i futuri elementi che terraform creerà:
    ```console
    user@user:~$ ./terraform plan
    ```
  
    - Avviare la creazione dell'infrastruttura:
    ```console
    user@user:~$ ./terraform apply -auto-approve
    ```
    
    N.B.: In caso di errore di connessione ssh eseguire ./terraform destroy e cancellare il security group recentemente creato su AWS. Rieseguire quindi il comando di "./terraform apply"

- Attendere che Terraform completi la creazione dell'infrastruttura. I tempi di attesa si attestano attorno ai 40 minuti data la presenza di file di dimensioni superiori al gigabyte da trasferire
- Connettersi ad ogni macchina, sia master che slave:
    - Accedere su AWS a "EC2"
    - Cliccare su "Istanze", dovrebbero essere presenti 7 macchine in stato "running" con nome "namenode" e "slave1", ... ,"slave6"
    - Per ogni macchina, aprire un terminale alla posizione in cui è salvata la chiave PEM
    - Su AWS, clic destro sul nome della macchina e selezionare "Connect" e poi il menù a tendina "SSH client". Copiare la stringa sotto "Example"
    - Incollare la stringa su terminale ed eseguire (quando richiesto, digitare "yes")
    - Verificare (nel master) che siano presenti le cartelle "hadoop" e "spark", e 3 file con estensione ".py"
    - Verificare (negli slave) che siano presenti le cartelle "hadoop" e "spark", e 1 file con estensione ".py"
    - Eseguire il comando su tutte le macchine:
    ```console
    sudo apt-get install openjdk-8-jdk
    ```
    L'outuput dovrebbe affermare che le "openjdk" sono state già installate. Se dovesse iniziarne l'installazione, portarla a termine.

GUIDA ALL'ESECUZIONE
 
- Avviare il master dalla macchina "namenode" eseguendo il seguente comando dalla cartella home:

    ```console
    user@user:~$ ./spark/sbin/start-master.sh
    ```
- Accedere su browser all'indirizzo che si trova su AWS alla voce "Public IPv4 DNS" dell'istanza master, aggiungendo alla fine la porta 8080:

```console
PUBLICDNS:8080
```
Attendere anche una decina di secondi, se i comandi sono stati eseguiti correttamente allora si vedrà a schermo l'interfaccia di Spark
- Avviare il numero di slave che si preferisce da datanode1, ..., datanode6:

    ```console
    user@user:~$ ./spark/sbin/start-slave.sh spark://PUBLICDNS:7077
    ```
- Ricaricare la pagina "PUBLICDNS:8080" e verificare che sia attivo il numero di slave desiderato
- Per eseguire il progetto, andare nella cartella home del master ed eseguire da terminale:

```console
./spark/bin/spark-submit --master spark://PUBLICDNS:7077 create_lexicons_sprk.py
```
Se i comandi sono stati impartiti correttamente, il programma comincerà a mostrare i suoi output.

N.B.: Se si vuole ripetere l'esecuzione del programma senza modificarne i parametri, eliminare la cartella creata all'interno di "sprk_lexicons/3 classes", il cui formato dovrebbe essere simile a:

```console
Financial_only_abs_delta_lookback_28_ngrams_(1, 1)_stemming_True_remove_stopwords_True_max_df_0.9_min_df_10
```
