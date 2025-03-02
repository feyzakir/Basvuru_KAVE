import yfinance as yf
import streamlit as st

def yilsonu_getirisi(symbol, yil):
 
    baslangic_tarihi = f'{yil}-01-01'
    bitis_tarihi = f'{yil}-12-31'
    
   
    veri = yf.download(symbol, start=baslangic_tarihi, end=bitis_tarihi)
    
    baslangic_fiyati = veri['Close'].iloc[0]
    bitis_fiyati = veri['Close'].iloc[-1]
    
    return (bitis_fiyati - baslangic_fiyati) / baslangic_fiyati * 100


def altin_gumus_karsilastir(yil):
    altin_sembol = 'GLD'
    gumus_sembol = 'SLV'   

    altin_getirisi = float(yilsonu_getirisi(altin_sembol, yil))
    gumus_getirisi = float(yilsonu_getirisi(gumus_sembol, yil))

    if altin_getirisi > gumus_getirisi:
        return f"{yil} yılında Altın daha fazla kazandırdı: {altin_getirisi:.2f}%", "https://image.cnnturk.com/i/cnnturk/75/1200x675/652db00aa468600accbd2da4.jpg"
    elif gumus_getirisi > altin_getirisi:
        return f"{yil} yılında Gümüş daha fazla kazandırdı: {gumus_getirisi:.2f}%", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQuxJ-T1_W4Tjmx9h6DLbaYW6BlvU-_cPezyQ&s"
    else:
        return f"{yil} yılında Altın ve Gümüş eşit kazanç sağladı.", None


st.title('Altın ve Gümüş Kazanç Karşılaştırması🥇🥈')
yil = st.number_input("Karşılaştırmak istediğiniz yılı girin:", min_value=2006, max_value=2025, step=1)

if st.button("Karşılaştır🥇🥈"):
    sonuc, resim_url = altin_gumus_karsilastir(yil)
    st.write(sonuc)
    if resim_url:
        st.image(resim_url)

