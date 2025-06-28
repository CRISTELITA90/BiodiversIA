
# Script: comparacion_endemicos_canarias.R
# Objetivo: Comparar ocurrencias de especies endémicas canarias desde GBIF

# Instalar paquetes si no los tienes
# install.packages("rgbif")
# install.packages("ggplot2")
# install.packages("leaflet")
# install.packages("dplyr")

library(rgbif)
library(ggplot2)
library(leaflet)
library(dplyr)

#  Lista de especies endémicas a comparar
especies <- c(
  "Gallotia galloti",
  "Gallotia stehlini",
  "Chalcides viridanus",
  "Tarentola angustimentalis",
  "Pericallis hadrosoma"
)

#  Descargar datos para cada especie
todas_ocurrencias <- data.frame()

for (esp in especies) {
  res <- occ_search(
    scientificName = esp,
    country = "ES",
    hasCoordinate = TRUE,
    limit = 2000
  )

  if (!is.null(res$data)) {
    df <- res$data
    df$especie <- esp
    todas_ocurrencias <- rbind(todas_ocurrencias, df)
  }
}

#  LIMPIEZA
datos_filtrados <- todas_ocurrencias %>%
  filter(!is.na(year), !is.na(decimalLatitude), !is.na(decimalLongitude))

#  GRÁFICO DE BARRAS: registros por especie
ggplot(datos_filtrados, aes(x = especie)) +
  geom_bar(fill = "darkgreen") +
  labs(title = "Total de registros por especie endémica",
       x = "Especie",
       y = "Nº de registros") +
  theme_minimal()

#  GRÁFICO TEMPORAL
ggplot(datos_filtrados, aes(x = year, fill = especie)) +
  geom_bar(position = "dodge") +
  labs(title = "Registros por año y especie",
       x = "Año",
       y = "Nº de registros") +
  theme_minimal()

# 🗺️ MAPA DE OCURRENCIAS
leaflet(datos_filtrados) %>%
  addTiles() %>%
  addCircleMarkers(
    ~decimalLongitude, ~decimalLatitude,
    color = ~as.factor(especie),
    popup = ~paste(especie, "<br>Año:", year),
    radius = 3,
    fillOpacity = 0.7
  ) %>%
  addLegend("bottomright", pal = colorFactor(rainbow(length(especies)), especies), values = ~especie) %>%
  setView(lng = -15.5, lat = 28.1, zoom = 7)

#  Exportar datos
write.csv(datos_filtrados, "registros_endemicos_canarias.csv", row.names = FALSE)
