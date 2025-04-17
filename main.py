from generarDataset import generar_dataset
from arbolDecision import id3
from graficadorArbol import dibujar_arbol
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QGridLayout, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QHBoxLayout, QScrollArea
from PyQt6.QtGui import QPixmap, QBrush, QPen
from PyQt6.QtCore import Qt, QRectF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import math
from sklearn import tree
import cairosvg  # Importar cairosvg para la conversión de SVG a PNG

class Main(QMainWindow):
    def __init__(self):
        super().__init__()

        # Paso 1: Generar datos
        print("Generando dataset...")
        df = generar_dataset()
        self.dfGlobal = df
        self.dataset = df  # Guardar el dataset original para usarlo en dibujar_arbol
        if df is None or df.empty:
            print("El dataset generado está vacío o es inválido.")
            sys.exit(1)
        else:
            print("listo\n")

        # Paso 2: Visualizar entropía del nodo raíz
        print("Mostrando entropías...")
        print("omitiendo...")
        #mostrar_bolas_por_clase(df)
        print("listo\n")

        # Paso 3: Crear árbol
        print("Creando árbol...")
        atributos = ['Color', 'Altura']
        try:
            arbol = id3(df, atributos)
        except Exception as e:
            print(f"Error al crear el árbol: {e}")
            sys.exit(1)
        print("listo\n")

        # Paso 4: Navegar por árbol
        #navegar_arbol(arbol)
        print("Iniciando interfaz...")

        self.arbol = arbol
        self.historial = []
        self.nodo_actual = arbol

        # Configuración de la ventana principal
        self.setWindowTitle("Navegador de Árbol")

        # Crear el diseño principal como una cuadrícula
        self.grid_layout = QGridLayout()

        # Panel 1: Interfaz existente
        self.panel1 = QWidget()
        self.panel1_layout = QVBoxLayout()
        self.panel1.setLayout(self.panel1_layout)

        self.boton_avanzar = QPushButton("Avanzar")
        self.boton_avanzar.clicked.connect(self.avanzar)
        self.panel1_layout.addWidget(self.boton_avanzar)

        self.boton_retroceder = QPushButton("Retroceder")
        self.boton_retroceder.clicked.connect(self.retroceder)
        self.panel1_layout.addWidget(self.boton_retroceder)

        self.boton_salir = QPushButton("Salir")
        self.boton_salir.clicked.connect(self.salir)
        self.panel1_layout.addWidget(self.boton_salir)

        self.label_diagrama = QLabel()
        self.panel1_layout.addWidget(self.label_diagrama)

        # Panel 2: Diagrama de dispersión
        self.panel2 = QWidget()
        self.panel2_layout = QVBoxLayout()
        self.panel2.setLayout(self.panel2_layout)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.panel2_layout.addWidget(self.canvas)

        # Panel 3: Pruebas de entropía (Interfaz de EntropyVisualizer)
        self.panel3 = QWidget()
        self.panel3_layout = QVBoxLayout()
        self.panel3.setLayout(self.panel3_layout)

        self.num_positive = 3
        self.num_negative = 3

        self.label_info = QLabel()
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        self.update_entropy_label()
        self.plot_entropy_curve()

        # Botones
        self.btn_add_pos = QPushButton("Añadir Positivo")
        self.btn_remove_pos = QPushButton("Eliminar Positivo")
        self.btn_add_neg = QPushButton("Añadir Negativo")
        self.btn_remove_neg = QPushButton("Eliminar Negativo")

        self.btn_add_pos.clicked.connect(lambda: self.modify_entropy("pos", 1))
        self.btn_remove_pos.clicked.connect(lambda: self.modify_entropy("pos", -1))
        self.btn_add_neg.clicked.connect(lambda: self.modify_entropy("neg", 1))
        self.btn_remove_neg.clicked.connect(lambda: self.modify_entropy("neg", -1))

        # Diseños
        self.panel3_layout.addWidget(self.label_info)
        self.panel3_layout.addWidget(self.view)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_add_neg)
        btn_layout.addWidget(self.btn_remove_neg)
        btn_layout.addWidget(self.btn_add_pos)
        btn_layout.addWidget(self.btn_remove_pos)

        self.panel3_layout.addLayout(btn_layout)

        # Panel 4: Mostrar la imagen del árbol de decisión
        self.panel4 = QWidget()
        self.panel4_layout = QVBoxLayout()
        self.panel4.setLayout(self.panel4_layout)

        self.label_arbol = QLabel("Árbol de Decisión (SVG Generado)")
        self.panel4_layout.addWidget(self.label_arbol)

        self.svg_widget = QLabel()
        self.panel4_layout.addWidget(self.svg_widget)

        # Agregar los paneles al diseño de la cuadrícula
        self.grid_layout.addWidget(self.panel1, 0, 0)  # Panel 1 en la esquina superior izquierda
        self.grid_layout.addWidget(self.panel2, 0, 1)  # Panel 2 en la esquina superior derecha
        self.grid_layout.addWidget(self.panel3, 1, 0)  # Panel 3 en la esquina inferior izquierda
        self.grid_layout.addWidget(self.panel4, 1, 1)  # Panel 4 en la esquina inferior derecha

        # Ajustar proporciones de las filas y columnas
        self.grid_layout.setRowStretch(0, 1)  # Primera fila ocupa 50% del espacio vertical
        self.grid_layout.setRowStretch(1, 1)  # Segunda fila ocupa 50% del espacio vertical
        self.grid_layout.setColumnStretch(0, 1)  # Primera columna ocupa 50% del espacio horizontal
        self.grid_layout.setColumnStretch(1, 1)  # Segunda columna ocupa 50% del espacio horizontal


        self.generar_y_mostrar_arbol_svg()

        # Crear un widget central para contener el diseño de la cuadrícula
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.grid_layout)

        # Crear un área de scroll y agregar el widget central
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)  # Hacer que el área de scroll sea redimensionable
        self.scroll_area.setWidget(self.central_widget)

        # Establecer el área de scroll como el widget central de la ventana principal
        self.setCentralWidget(self.scroll_area)

        self.actualizar_interfaz()
        print("listo\n")

    def actualizar_interfaz(self):
        if 'atributo' in self.nodo_actual:
            self.boton_avanzar.setEnabled(True)
        else:
            self.boton_avanzar.setEnabled(False)
        self.actualizar_diagrama()
        self.actualizar_dispersión()

    def actualizar_diagrama(self):
        try:
            # Generar el diagrama del árbol mostrando todas las ramas desde la raíz hasta el nivel actual
            nivel_actual = len(self.historial) + 1  # Nivel actual basado en el historial
            dibujar_arbol(self.dataset, nivel=nivel_actual, nombre_archivo='arbol_nivel_actual')

            # Cargar la imagen generada en el QLabel
            pixmap = QPixmap('arbol_nivel_actual.png')

            # Redimensionar la imagen para que se ajuste al ancho del panel
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.label_diagrama.width(),  # Ancho del QLabel
                    self.label_diagrama.height(),  # Alto del QLabel
                    aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio  # Mantener proporciones
                )
                self.label_diagrama.setPixmap(scaled_pixmap)
            else:
                print("Error: No se pudo cargar la imagen del árbol.")
        except Exception as e:
            print(f"Error al generar el diagrama del árbol: {e}")

    def actualizar_dispersión(self):
        try:
            # Limpiar la figura existente
            self.figure.clear()

            # Obtener los datos del dataset
            df = self.dfGlobal  # Dataset original
            if df is not None and not df.empty:
                atributos = df.columns[:-1]  # Usar todos los atributos excepto la clase
                clase_columna = df.columns[-1]  # Última columna como clase
                x = df[atributos[0]]  # Primer atributo como eje X
                y = df[atributos[1]]  # Segundo atributo como eje Y
                clases = df[clase_columna].unique()  # Clases únicas

                # Codificar las variables categóricas
                le_x = LabelEncoder()
                le_y = LabelEncoder()
                le_clase = LabelEncoder()
                df[atributos[0]] = le_x.fit_transform(df[atributos[0]])
                df[atributos[1]] = le_y.fit_transform(df[atributos[1]])
                df[clase_columna] = le_clase.fit_transform(df[clase_columna])

                # Entrenar un árbol de decisiones con profundidad limitada al nivel actual
                X = df[[atributos[0], atributos[1]]]
                y = df[clase_columna]
                nivel_actual = len(self.historial) + 1  # Nivel actual basado en el historial
                clf = DecisionTreeClassifier(criterion='entropy', max_depth=nivel_actual, random_state=42)
                clf.fit(X, y)

                # Crear una cuadrícula de puntos para predecir las clases
                x_min, x_max = X[atributos[0]].min() - 1, X[atributos[0]].max() + 1
                y_min, y_max = X[atributos[1]].min() - 1, X[atributos[1]].max() + 1
                xx, yy = np.meshgrid(
                    np.arange(x_min, x_max, 0.1),
                    np.arange(y_min, y_max, 0.1)
                )
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                # Crear el diagrama de dispersión con áreas de decisión
                ax = self.figure.add_subplot(111)
                ax.contourf(xx, yy, Z, alpha=0.3, cmap='Pastel1')  # Colorear áreas de decisión

                # Dibujar los puntos del dataset
                for clase in clases:
                    puntos_clase = df[df[clase_columna] == le_clase.transform([clase])[0]]
                    ax.scatter(
                        puntos_clase[atributos[0]],
                        puntos_clase[atributos[1]],
                        label=f"Clase {clase}",
                        alpha=0.7
                    )

                # Etiquetas y leyenda
                ax.set_xlabel(atributos[0])
                ax.set_ylabel(atributos[1])
                ax.set_title(f"Diagrama de Dispersión (Nivel: {nivel_actual})")
                ax.legend()

                # Actualizar el canvas
                self.canvas.draw()
            else:
                print("Error: El dataset está vacío o no es válido.")
        except Exception as e:
            print(f"Error al generar el diagrama de dispersión: {e}")

    def modify_entropy(self, tipo, delta):
        if tipo == "pos":
            self.num_positive = max(0, self.num_positive + delta)
        else:
            self.num_negative = max(0, self.num_negative + delta)

        self.update_entropy_label()
        self.plot_entropy_curve()

    def calcular_entropia(self, p, n):
        total = p + n
        if total == 0:
            return 0
        prob_p = p / total
        prob_n = n / total
        ent = 0
        for prob in (prob_p, prob_n):
            if prob > 0:
                ent -= prob * math.log2(prob)
        return ent

    def update_entropy_label(self):
        entropy = self.calcular_entropia(self.num_positive, self.num_negative)
        self.label_info.setText(
            f"<b>Entropía:</b> {entropy:.3f}<br>"
            f"<b>Positivo:</b> {self.num_positive} &nbsp; "
            f"<b>Negativo:</b> {self.num_negative}"
        )

    def plot_entropy_curve(self):
        self.scene.clear()

        # Dibujar curva de entropía
        width, height = 600, 200
        margin = 50
        for x in range(101):
            p = x / 100
            n = 1 - p
            if p == 0 or p == 1:
                y = 0
            else:
                y = -p * math.log2(p) - n * math.log2(n)
            x_pos = margin + x * (width / 100)
            y_pos = margin + (1 - y) * height
            self.scene.addEllipse(QRectF(x_pos, y_pos, 5, 5), QPen(), QBrush(Qt.GlobalColor.white))

        # Dibujar "recipiente" con bolas de colores
        total = self.num_positive + self.num_negative
        if total == 0:
            prop = 0
        else:
            prop = self.num_positive / total
        pos_x = margin + prop * width
        pos_y = margin + (1 - self.calcular_entropia(self.num_positive, self.num_negative)) * height

        # Círculo recipiente
        container = QGraphicsEllipseItem(pos_x - 25, pos_y - 25, 50, 50)
        container.setPen(QPen(Qt.GlobalColor.white, 2))
        self.scene.addItem(container)

        # Bolas dentro del círculo
        radius = 10
        spacing = 10
        offset_x = pos_x - 25
        offset_y = pos_y - 25
        index = 0
        for _ in range(self.num_positive):
            x = offset_x + (index % 5) * spacing
            y = offset_y + (index // 5) * spacing
            self.scene.addEllipse(QRectF(x, y, radius, radius),
                                  QPen(), QBrush(Qt.GlobalColor.cyan))
            index += 1
        for _ in range(self.num_negative):
            x = offset_x + (index % 5) * spacing
            y = offset_y + (index // 5) * spacing
            self.scene.addEllipse(QRectF(x, y, radius, radius),
                                  QPen(), QBrush(Qt.GlobalColor.red))
            index += 1

        self.view.setScene(self.scene)

    def reconstruir_camino(self):
        # Reconstruir el camino desde la raíz hasta el nodo actual
        camino = self.historial + [self.nodo_actual]
        return camino

    def avanzar(self):
        if 'atributo' in self.nodo_actual:
            # Avanzar al siguiente nivel sin necesidad de seleccionar un camino
            self.historial.append(self.nodo_actual)
            self.actualizar_interfaz()  # Actualizar todo, incluyendo el diagrama de dispersión

    def retroceder(self):
        if self.historial:
            # Retroceder al nivel anterior
            self.historial.pop()
            self.actualizar_interfaz()  # Actualizar todo, incluyendo el diagrama de dispersión

    def salir(self):
        print("Saliendo de la aplicación...")
        self.close()

    def generar_y_mostrar_arbol_svg(self):
        try:
            dataset = self.dfGlobal
            print("Generated dataset:\n", dataset)

            # Codificar las variables categóricas
            label_encoders = {}
            for column in dataset.columns:
                le = LabelEncoder()
                dataset[column] = le.fit_transform(dataset[column])
                label_encoders[column] = le

            X = dataset.iloc[:, :-1]  # Características
            y = dataset.iloc[:, -1]   # Etiqueta
            print("Encoded X:\n", X)
            print("Encoded y:\n", y)

            # Entrenar el clasificador con hiperparámetros predeterminados
            print("Creando clf...")
            clf = DecisionTreeClassifier(random_state=1234)
            model = clf.fit(X, y)
            print("Creando model...")

            text_representation = tree.export_text(clf, feature_names=X.columns.astype(str))
            print("Text_representation:\n", text_representation)

            import dtreeviz
            print("Creando viz...")
            viz = dtreeviz.model(clf, X, y,
                            target_name="target",
                            feature_names=X.columns.astype(str),
                            class_names=[str(cls) for cls in label_encoders[dataset.columns[-1]].classes_])
            print("Creando V...")
            v = viz.view()
            print("Guardando SVG...")
            v.save("decision_tree.svg")
            print("SVG del árbol de decisión generado como 'decision_tree.svg'.")

            # Convertir SVG a PNG
            cairosvg.svg2png(url="decision_tree.svg", write_to="decision_tree.png")
            print("SVG convertido a PNG: 'decision_tree.png'.")

            # Cargar la imagen generada en el QLabel
            pixmap = QPixmap('decision_tree.png')

            # Redimensionar la imagen para que se ajuste al ancho del panel
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.svg_widget.width(),  # Ancho del QLabel
                    self.svg_widget.height(),  # Alto del QLabel
                    aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio  # Mantener proporciones
                )
                self.svg_widget.setPixmap(scaled_pixmap)
            else:
                print("Error: No se pudo cargar la imagen del árbol.")
        except Exception as e:
            print(f"Error al generar o mostrar el árbol de decisión en SVG: {e}")

app = QApplication(sys.argv)
ventana = Main()
ventana.showMaximized()

sys.exit(app.exec())