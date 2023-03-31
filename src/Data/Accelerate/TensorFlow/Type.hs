{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE UndecidableSuperClasses #-}

module Data.Accelerate.TensorFlow.Type where

import Data.Array.Accelerate.Representation.Shape hiding (union)
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.Array.Unique (UniqueArray(UniqueArray), withUniqueArrayPtr)
import Data.Array.Accelerate.Lifetime
import GHC.ForeignPtr
import Control.Monad.IO.Class (liftIO)
import Data.Array.Accelerate.Type (ScalarType (..), SingleType (NumSingleType), NumType (IntegralNumType, FloatingNumType), IntegralType (..), FloatingType (..), VectorType (..))

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Session                                 as TF
import qualified TensorFlow.Internal.FFI as FFI
import qualified Data.Vector as V
import Data.Int
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape, placeholder)
import Foreign (Ptr, castPtr, Word8)
import Unsafe.Coerce (unsafeCoerce)
import Data.Array.Accelerate.Analysis.Match ( type (:~:)(Refl), matchScalarType )
import qualified Data.Functor.Identity as TF
import qualified Data.Vector.Storable as S
import Text.Printf
import qualified Data.Complex
import qualified Data.Word
import qualified Data.ByteString
import Data.Array.Accelerate.Representation.Type (TypeR)

import Data.Type.Equality ((:~:)(..))
import Data.Proxy (Proxy(..))
import Data.Kind (Type)
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}

import GHC.TypeLits
import Data.Kind
import Data.Proxy
import Data.Type.Bool (If)

type TFAll = '[Data.Complex.Complex Double,
               Data.Complex.Complex Float,
               Bool,
               Data.Int.Int16,
               Data.Int.Int32,
               Data.Int.Int64,
               Data.Int.Int8,
               Data.Word.Word16,
               Data.Word.Word32,
               Data.Word.Word64,
               Data.Word.Word8,
               Double,
               Float]

type TFNum = TFAll TF.\\ '[Bool]

type TFNeg = TFNum TF.\\ '[Data.Word.Word8]
type TFSign = TFNeg

type TFTruncateMod = '[Data.Int.Int32, 
                       Data.Int.Int64,
                       Data.Word.Word16, 
                       Double,
                       Float]

type TFAbs = TFNum TF.\\ '[Data.Word.Word8, Data.Complex.Complex Double, Data.Complex.Complex Float]

data TensorDict types t where
  TensorDict :: TF.OneOf types t => TensorDict types t

type TensorType a = (S.Storable a, TF.TensorDataType S.Vector a, TF.TensorType a)

data TensorTypeDict a where
  TensorTypeDict :: TensorType a => TensorTypeDict a

tfAllDict :: ScalarType a -> TensorDict TFAll a
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt)))    = error "not a TF all type"
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt8)))   = TensorDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt16)))  = TensorDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt32)))  = TensorDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt64)))  = TensorDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord)))   = error "not a TF all type"
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord8)))  = TensorDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord16))) = TensorDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord32))) = TensorDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord64))) = TensorDict
tfAllDict (SingleScalarType (NumSingleType (FloatingNumType TypeHalf)))   = error "not a TF all type"
tfAllDict (SingleScalarType (NumSingleType (FloatingNumType TypeFloat)))  = TensorDict
tfAllDict (SingleScalarType (NumSingleType (FloatingNumType TypeDouble))) = TensorDict
tfAllDict (VectorScalarType _)                                            = error "not a TF all type"

tfNumDict :: NumType a -> TensorDict TFNum a
tfNumDict (IntegralNumType TypeInt)    = error "not a TF num type"
tfNumDict (IntegralNumType TypeInt8)   = TensorDict
tfNumDict (IntegralNumType TypeInt16)  = TensorDict
tfNumDict (IntegralNumType TypeInt32)  = TensorDict
tfNumDict (IntegralNumType TypeInt64)  = TensorDict
tfNumDict (IntegralNumType TypeWord)   = error "not a TF num type"
tfNumDict (IntegralNumType TypeWord8)  = TensorDict
tfNumDict (IntegralNumType TypeWord16) = TensorDict
tfNumDict (IntegralNumType TypeWord32) = TensorDict
tfNumDict (IntegralNumType TypeWord64) = TensorDict
tfNumDict (FloatingNumType TypeHalf)   = error "not a TF num type"
tfNumDict (FloatingNumType TypeFloat)  = TensorDict
tfNumDict (FloatingNumType TypeDouble) = TensorDict

tfNegDict :: NumType a -> TensorDict TFNeg a
tfNegDict (IntegralNumType TypeWord8)  = error "not a TF neg type"
tfNegDict (IntegralNumType TypeInt)    = error "not a TF neg type"
tfNegDict (IntegralNumType TypeInt8)   = TensorDict
tfNegDict (IntegralNumType TypeInt16)  = TensorDict
tfNegDict (IntegralNumType TypeInt32)  = TensorDict
tfNegDict (IntegralNumType TypeInt64)  = TensorDict
tfNegDict (IntegralNumType TypeWord)   = error "not a TF neg type"
tfNegDict (IntegralNumType TypeWord16) = TensorDict
tfNegDict (IntegralNumType TypeWord32) = TensorDict
tfNegDict (IntegralNumType TypeWord64) = TensorDict
tfNegDict (FloatingNumType TypeHalf)   = error "not a TF neg type"
tfNegDict (FloatingNumType TypeFloat)  = TensorDict
tfNegDict (FloatingNumType TypeDouble) = TensorDict

tfTruncateModDict :: NumType a -> TensorDict TFTruncateMod a
tfTruncateModDict (IntegralNumType TypeWord8)   = error "not a TF truncate mod type"
tfTruncateModDict (IntegralNumType TypeInt)     = error "not a TF truncate mod type"
tfTruncateModDict (IntegralNumType TypeInt8)    = error "not a TF truncate mod type"
tfTruncateModDict (IntegralNumType TypeInt16)   = error "not a TF truncate mod type"
tfTruncateModDict (IntegralNumType TypeInt32)   = TensorDict
tfTruncateModDict (IntegralNumType TypeInt64)   = TensorDict
tfTruncateModDict (IntegralNumType TypeWord)    = error  "not a TF truncate mod type"
tfTruncateModDict (IntegralNumType TypeWord16)  = TensorDict
tfTruncateModDict (IntegralNumType TypeWord32)  = TensorDict
tfTruncateModDict (IntegralNumType TypeWord64)  = TensorDict
tfTruncateModDict (FloatingNumType TypeHalf)    = error "not a TF truncate mod type"
tfTruncateModDict (FloatingNumType TypeFloat)   = TensorDict
tfTruncateModDict (FloatingNumType TypeDouble)  = TensorDict

tensorTypeDict :: ScalarType a -> TensorTypeDict a
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt)))    = error "not a tensortype"
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt8)))   = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt16)))  = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt32)))  = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt64)))  = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord)))   = error "not a tensortype"
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord8)))  = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord16))) = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord32))) = error "not a tensortype"
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord64))) = error "not a tensortype"
tensorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeHalf)))   = error "not a tensortype"
tensorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeFloat)))  = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeDouble))) = TensorTypeDict
tensorTypeDict (VectorScalarType _)                                            = error "not a tensortype"